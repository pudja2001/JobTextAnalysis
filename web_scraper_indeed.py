import csv
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import re

def generate_url(position, location):
    """Generate URL from position and location"""
    template = "https://id.indeed.com/jobs?q={}&l={}"
    url = template.format(position, location)
    return url

def get_record(curr_card):
    """Extract the job data from single card"""
    atag = curr_card.h2.a
    # Extract the job title from html
    job_title = atag.span.get("title")
    # Extract job url
    job_url = "https://www.indeed.com" + str(atag.get("href"))
    # Extract Job Company
    job_company = curr_card.find("span", "companyName").text
    # Extract Job Location
    job_location = curr_card.find("div", "companyLocation").text
    # Extract Job Summary
    job_summary = curr_card.find("div", "job-snippet").text
    job_summary = re.sub('\n', '', job_summary)
    # Extract Posting Date
    posting_date = curr_card.find("span", "date").text
    posting_date = re.sub('Posted', '', posting_date)
    posting_date = re.sub('EmployerAktif ', '', posting_date)
    today_date = datetime.today().strftime('%d-%m-%Y')
    # Scrape job description on each job page
    job_response = requests.get(job_url)
    job_soup = BeautifulSoup(job_response.text, 'html.parser')
    job_desc = job_soup.find("div", "jobsearch-jobDescriptionText").text
    # Extract Job Salary
    try:
        job_salary = curr_card.find("div", "metadata salary-snippet-container").text
    except AttributeError:
        job_salary = ''
        
    record = (job_title, job_company, job_location, posting_date, today_date, job_summary, job_desc, job_salary)
    
    return record

def main(positions, locations, csv_name):
    """run the main program"""

    records = []
    
    for position in positions:
        for location in locations:
            # Get job URL page based on position and location  
            url = generate_url(position, location)

            # Extract the Job Data
            while True:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                cards = soup.find_all("div", "cardOutline")

                for card in cards:
                    record = get_record(card)
                    records.append(record)

                try:
                    url = "https://id.indeed.com/" + soup.find("a", {"aria-label":"Berikutnya"}).get("href")
                except AttributeError:
                    break

    # Save the Job Data
    with open(csv_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['JobTitle', 'Company', 'Location', 'PostDate', 'ScrapeDate', 'JobSummary', 'JobDesc', 'Salary'])
        writer.writerows(records)

if __name__ == "__main__":
    positions = ["data scientist", "data analyst", "business intelligence", "data engineer"]
    locations = ["jakarta"]
    main(positions, locations, 'datajobs.csv')