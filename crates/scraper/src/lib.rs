use rayon::prelude::*;
use regex::Regex;
use reqwest::Url;
use select::document::Document;
use select::predicate::Class;
use select::predicate::Name;
use select::predicate::Predicate;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::Error as IoErr;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize)]
struct Cache {
    downloaded_bills: HashSet<String>,
}

#[derive(Debug)]
pub enum Error {
    Write { url: String, e: IoErr },
    Fetch { url: String, e: reqwest::Error },
}

pub type Result<T> = std::result::Result<T, Error>;

impl<S: AsRef<str>> From<(S, IoErr)> for Error {
    fn from((url, e): (S, IoErr)) -> Self {
        Error::Write {
            url: url.as_ref().to_string(),
            e,
        }
    }
}

impl<S: AsRef<str>> From<(S, reqwest::Error)> for Error {
    fn from((url, e): (S, reqwest::Error)) -> Self {
        Error::Fetch {
            url: url.as_ref().to_string(),
            e,
        }
    }
}

pub fn scrape(output_path: &str, cache_path: &str) -> Result<()> {
    // Get beginning time for timing the call
    let now = Instant::now();

    // Get list of previously downloaded bills from the cache
    // or make it if it doesn't exist
    let mut cache = read_cache(cache_path);

    // Pull text from the congress search page
    let client = reqwest::blocking::Client::new();
    let origin_url ="https://congress.gov/search?q=%7B%22source%22%3A%22legislation%22%2C%22congress%22%3A%22117%22%2C%22type%22%3A%22bills%22%2C%22chamber%22%3A%22House%22%7D&searchResultViewType=expanded&pageSort=documentNumber%3Aasc&pageSize=100&page=1";
    println!("Fetching first search page");
    let body = fetch_url(&client, origin_url)?;

    // Create a hash set for visited pages, so we don't revisit them
    let mut visited = HashSet::new();
    // List of potential bills
    let valid_bills = Arc::new(Mutex::new(HashSet::new()));
    // List of bills that are already downloaded
    let downloaded_bills = Arc::new(Mutex::new(HashSet::new()));

    // Potential pages to visit with more bills
    let mut valid_search_pages = HashSet::new();

    // We already visited the origin page
    visited.insert(origin_url.to_string());

    // Extract the links from the page, any that ends with a page number
    // is a new search page we can visit to get more bills
    let found_urls = get_links_from_html(&body);
    for i in found_urls {
        let v: Vec<&str> = i.split('&').collect();
        let extension = v.last().unwrap();
        if extension.starts_with("page=") {
            valid_search_pages.insert(i.to_string());
        }
    }

    // Get the bills on the page to check for text
    let bill_numbers_on_page = get_numbers_from_html(&body);

    // We need to compare them against the cache to make sure
    // we don't try to download one that's already present
    let cache_bills = &cache.downloaded_bills.clone();

    // If any of the bills aren't in the cache, we'll put them
    // in the valid bills list
    for i in bill_numbers_on_page.difference(cache_bills) {
        valid_bills.lock().unwrap().insert(i.clone());
    }

    // Get the bill pages we haven't visited yet
    let mut new_urls = valid_search_pages
        .difference(&visited)
        .map(|x| x.to_string())
        .collect::<HashSet<String>>();

    // While we still have new urls, extract the bill numbers and search pages
    // on that page
    while !new_urls.is_empty() {
        let (found_urls, errors): (Vec<Result<HashSet<String>>>, Vec<_>) = new_urls
            .par_iter()
            .map(|url| -> Result<HashSet<String>> {
                println!("Searching {}", url);
                let body = fetch_url(&client, url)?;

                let bill_numbers_on_page = get_numbers_from_html(&body);
                for i in bill_numbers_on_page.difference(cache_bills) {
                    valid_bills.lock().unwrap().insert(i.clone());
                }
                let mut search_pages = HashSet::new();
                let found_urls = get_links_from_html(&body);
                for i in found_urls {
                    let v: Vec<&str> = i.split('&').collect();
                    let extension = v.last().unwrap();
                    if extension.starts_with("page=") {
                        search_pages.insert(i.to_string());
                    }
                }
                Ok(search_pages)
            })
            .partition(Result::is_ok);

        // Add those urls to the visited page list so we don't visit them again
        visited.extend(new_urls);

        // Reset new_urls with all of the urls we haven't visited
        new_urls = found_urls
            .into_par_iter()
            .map(Result::unwrap)
            .reduce(HashSet::new, |mut acc, x| {
                acc.extend(x);
                acc
            })
            .difference(&visited)
            .map(|x| x.to_string())
            .collect::<HashSet<String>>();
        println!(
            "Errors: {:#?}",
            errors
                .into_iter()
                .map(Result::unwrap_err)
                .collect::<Vec<Error>>()
        )
    }

    // Iterate through the bills, extracting the text
    for i in valid_bills.lock().iter() {
        let bills_list = i.iter().into_iter().collect::<Vec<_>>();
        let (_, _): (Vec<Result<String>>, Vec<_>) = bills_list
            .par_iter()
            .map(|url| -> Result<String> {
                // Fetch the bill
                let body = fetch_url(&client, &get_bill_text_url(url))?;
                let text = get_bill_text_from_html(&body);
                // We don't write bills that are too short, as they are usually
                // just the warning that the bill will be summarized soon
                if text.len() > 1000 {
                    write_file(output_path, url, &text)?;
                    downloaded_bills.lock().unwrap().insert(url.clone());
                    println!("Downloaded {}", url.to_string());
                }
                Ok(url.to_string())
            })
            .partition(Result::is_ok);
        println!(
            "Downloaded bills: {}",
            downloaded_bills.lock().unwrap().len()
        );
        for i in downloaded_bills.lock().unwrap().drain() {
            cache.downloaded_bills.insert(i.to_string());
        }
        let json = serde_json::to_string_pretty(&cache).unwrap();
        fs::write(cache_path, json).map_err(|e| ("cache.json", e))?;
    }
    println!("Elapsed time: {}", now.elapsed().as_secs());
    Ok(())
}

fn get_links_from_html(html: &str) -> HashSet<String> {
    Document::from(html)
        .find(Name("a").or(Name("link")))
        .filter_map(|n| n.attr("href"))
        .filter(has_extension)
        .filter_map(normalize_url)
        .collect::<HashSet<String>>()
}

fn get_numbers_from_html(html: &str) -> HashSet<String> {
    Document::from(html)
        .find(Name("a").or(Name("link")))
        .filter_map(|n| n.attr("href"))
        .filter(has_extension)
        .filter_map(extract_bill_number)
        .collect::<HashSet<String>>()
}

fn get_bill_text_from_html(html: &str) -> String {
    let document = Document::from(html);
    let mut docs = "".to_string();
    for node in document.find(Class("main-wrapper")) {
        let text = node.text();
        let re = Regex::new(r"<all>|<DOC>").unwrap();
        let split: Vec<&str> = re.split(&text).collect();
        if split.len() == 3 {
            docs = split[1].to_string();
        } else {
            docs = text;
        }
    }
    docs
}

fn extract_bill_number(url: &str) -> Option<String> {
    let v: Vec<&str> = url.split('/').collect();
    if v.len() > 5 {
        let arg = v[4];
        if arg.parse::<f64>().is_ok() {
            Some(arg.to_string())
        } else {
            None
        }
    } else {
        None
    }
}
fn normalize_url(url: &str) -> Option<String> {
    let new_url = Url::parse(url);
    match new_url {
        Ok(new_url) => {
            if let Some("congress.gov") = new_url.host_str() {
                Some(url.to_string())
            } else {
                None
            }
        }
        Err(_e) => {
            // Relative urls are not parsed by Reqwest
            if url.starts_with('/') {
                Some(format!("https://congress.gov{}", url))
            } else {
                None
            }
        }
    }
}

fn read_cache(path: &str) -> Cache {
    let mut file = match File::open(&path) {
        Err(why) => {
            println!("couldn't open {}: {}", path, why);
            File::create(path).unwrap();
            File::open(path).unwrap()
        }
        Ok(file) => file,
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let cache: Cache = match serde_json::from_str(&contents) {
        Err(why) => {
            println!("couldn't read json from {}: {}", path, why);
            let set = HashSet::new();
            Cache {
                downloaded_bills: set,
            }
        }
        Ok(json_data) => json_data,
    };
    cache
}

fn fetch_url(client: &reqwest::blocking::Client, url: &str) -> Result<String> {
    let mut res = client.get(url).send().map_err(|e| (url, e))?;
    //println!("Status for {}: {}", url, res.status());

    let mut body = String::new();
    res.read_to_string(&mut body).map_err(|e| (url, e))?;
    Ok(body)
}

fn has_extension(url: &&str) -> bool {
    Path::new(&url).extension().is_none()
}

fn get_bill_text_url(bill_number: &str) -> String {
    let url = format!(
        "https://www.congress.gov/bill/117th-congress/house-bill/{}/text?format=txt&r=84&s=3",
        bill_number
    );
    url
}

fn write_file(root: &str, path: &str, content: &str) -> Result<()> {
    let index = format!("{}/{}_text.txt", root, path);
    fs::write(&index, content).map_err(|e| (&index, e))?;

    Ok(())
}
