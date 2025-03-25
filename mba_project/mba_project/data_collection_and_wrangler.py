import random
import argparse
import serpapi
import json
import time
import os
import pandas as pd
import openai
import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint
from faker import Faker
from openai import OpenAI
from serpapi import HTTPConnectionError
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("SERPAPI_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_host = os.getenv("POSTGRES_HOST")
db_port = os.getenv("POSTGRES_PORT")
db_name = os.getenv("POSTGRES_DB")

client = OpenAI()
client.api_key = openai_key

connection_string = (
    f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)
engine = create_engine(connection_string)

HOURS_PER_YEAR = 2080  # Standard full-time hours in a year


def insert_job_detail(job_id, salary_from, salary_to):
    with engine.connect() as conn:
        try:
            conn.execute(
                text(
                    """INSERT INTO job_detail
                    (job_id, salary_from, salary_to, parsed_skills)
                    VALUES (:job_id, :salary_from, :salary_to, NULL)"""
                ),
                {
                    "job_id": job_id,
                    "salary_from": salary_from,
                    "salary_to": salary_to
                },
            )
            conn.commit()
            print(f"Inserted job_id {job_id} into job_detail.")
        except Exception as e:
            print(f"Error inserting job_id {job_id}: {e}")


def update_job_skills(job_id, parsed_skills):
    with engine.connect() as conn:
        try:
            query = text(
                """
                UPDATE job_detail 
                SET parsed_skills = :parsed_skills 
                WHERE job_id = :job_id
            """
            )
            conn.execute(query, {
                "parsed_skills": parsed_skills,
                "job_id": job_id
            })
            conn.commit()
            print(f"Updated job_id {job_id} with skills: {parsed_skills}")
        except Exception as e:
            print(f"Error updating job_id {job_id}: {e}")


def get_job_ids():
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """SELECT DISTINCT r.job_id 
                 FROM raw_jobs r
                WHERE job_id NOT IN (
                       SELECT job_id FROM job_detail)"""
            )
        )
        job_ids = [row[0] for row in result]
    return job_ids


def get_unique_job_skills():
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """WITH job_skills AS (
                SELECT lower(TRIM(skills)) AS skill
                  FROM job_detail,
                UNNEST(STRING_TO_ARRAY(parsed_skills, ',')) AS skills
                )
                SELECT DISTINCT skill FROM job_skills;"""
            )
        )
        job_skills = [row[0] for row in result]
    return job_skills


def get_job_details():
    with engine.connect() as conn:
        query = """
        WITH
        distinct_jobs AS (
            SELECT DISTINCT
                job_id,
                skills
            FROM raw_jobs rj
            WHERE skills NOT IN ('{""}', '', 'React', 'Node.js', 'JavaScript',
                      'Java', 'SQL', 'Machine Learning',
                      'Artificial Intelligence',
                      'Python', 'Golang', 'Devops')
            AND skills IS NOT NULL
        )
        SELECT *
         FROM distinct_jobs
        WHERE job_id NOT IN (SELECT job_id FROM job_detail
                              WHERE parsed_skills IS NOT NULL);
        """
        result = conn.execute(text(query)).fetchall()
        return result


def insert_raw_jobs(df, table_name):
    try:
        # Write the DataFrame to a PostgreSQL table
        df.to_sql(table_name, engine, index=False, if_exists="append")
        print(f"Data successfully stored in the '{table_name}' table.")
    except Exception as e:
        print(f"Error occurred: {e}")


def extract_skills(message_content, retry_count):
    openai_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    client.api_key = openai_key

    
    
    # This is your prompt
    prompt = f"""Summarize the content of this message:

    {message_content}

    Return it in a string format with proper poctuation and line breaks.
    """
    # The return piece you can specify json, csv or other formats
     
    # The models get's timeout sometimes, so maybe use more than one model is a good idea.
    # If that was the case just add more models here.
    # The available model can be found here: https://openai.com/api/pricing/
    
    models = [
        "gpt-4o-mini",
    ]

    while retry_count < len(models):
        print(f"Retry Counter... {retry_count} \n")
        try:
            model = models[retry_count]
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        # This is the directions you give to GPT before you sent your prompt properly.
                        "content": """Your task is to create  
                        a summary of a given message capturing the context of it....""",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7, # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
                                 # If you decrease it too much it becames non-sense.
                max_tokens=80, # The maximum number of tokens that can be generated in the chat completion.
                top_p=1,
            )
            # Access the response content
            response_content = response.choices[0].message.content.strip()
            
            # This is just data treatment, maybe you can remove it for the use case
            # Remove triple backticks and the 'json' word if they exist
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            elif response_content.startswith("```"):
                response_content = response_content[3:-3].strip()

            # Now try to parse the cleaned response content as JSON
            try:
                skills_data = json.loads(response_content)
                skills_string = ", ".join(skills_data)
                return {"skills": skills_string, "rtc": retry_count}
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse the JSON output",
                    "rtc": retry_count
                }

        except openai.RateLimitError as e:
            print(f"Rate limit hit for {model}. Switching models... \n {e} \n")
            retry_count += 1
            # Wait before retrying to avoid hitting the limit again
            time.sleep(20)


def extract_qualifications(job_highlights):
    qualifications = []
    for highlight in job_highlights:
        if highlight.get("title") == "Qualifications":
            qualifications.extend(highlight.get("items", []))

    return ", ".join(qualifications)


def get_jobs(skill):
    jobs = []
    next_page_token = None

    client = serpapi.Client(api_key=api_key)
    while True:
        params = {
            "engine": "google_jobs",
            "q": f"{skill} Engineer",
            "gl": "us",
            "no_cache": "true",
            "ltype": 1,
            "num": 100,
        }

        if next_page_token:
            params["next_page_token"] = next_page_token

        try:
            # Make the API call within the loop
            results = client.search(params).as_dict()
        except HTTPConnectionError as e:
            print(f"Connection error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue  # Retry the current loop iteration
        except Exception as err:
            print(err)
            break

        job_results = results.get("jobs_results", [])

        if not job_results:
            break  # Exit the loop if no more jobs are found

        for job in job_results:
            job_id = job.get("job_id")
            if job_id:
                qualifications = skill

                if job.get("job_highlights", []):
                    qualifications = extract_qualifications(
                        job.get("job_highlights", [])
                    )

                jobs.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company_name": job.get("company_name"),
                        "description": job.get("description"),
                        "location": job.get("location"),
                        "via": job.get("via"),
                        "posted_at": job.get("detected_extensions", {}).get(
                            "posted_at"
                        ),
                        "skills": qualifications,
                    }
                )

        # Get the next page token from the current response
        next_page_token = results.get("serpapi_pagination", {}
                                      ).get("next_page_token")
        if not next_page_token:
            break  # Exit the loop if there's no next page

    df = pd.DataFrame(jobs)

    return df


def process_salary_data(salaries):
    hourly_salaries = []

    for salary in salaries:
        salary_from = salary.get("salary_from")
        salary_to = salary.get("salary_to")
        salary_periodicity = salary.get("salary_periodicity", "year")

        if salary_periodicity == "year":
            # Convert annual salary to hourly
            salary_from = salary_from / HOURS_PER_YEAR if salary_from else None
            salary_to = salary_to / HOURS_PER_YEAR if salary_to else None

        hourly_salaries.append((salary_from, salary_to))

    # Handle cases with multiple salary sources: take the average range
    if hourly_salaries:
        salary_from_avg = sum(s[0] for s in hourly_salaries if s[0]) / len(
            hourly_salaries
        )
        salary_to_avg = sum(s[1] for s in hourly_salaries if s[1]) / len(
            hourly_salaries
        )
        return round(salary_from_avg, 2), round(salary_to_avg, 2)

    return None, None


def get_salary_info(job_id):
    client = serpapi.Client(api_key=api_key)

    try:
        params = {
            "engine": "google_jobs_listing",
            "q": job_id,
        }
        result = client.search(params).as_dict()

        salaries = result.get("salaries", [])
        if not salaries:
            return None, None

        salary_from, salary_to = process_salary_data(salaries)

        return salary_from, salary_to

    except HTTPConnectionError as e:
        print(f"Connection error occurred: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        return get_salary_info(job_id)  # Retry the current job_id
    except Exception as err:
        print(f"Error occurred: {err}")
        return None, None


def get_job_salaries():
    job_ids = get_job_ids()

    for job_id in job_ids:
        salary_from, salary_to = get_salary_info(job_id)
        insert_job_detail(job_id, salary_from, salary_to)


def generate_weighted_talents(num_of_talents, starting_id, skills_with_weights):
    fake = Faker()
    talents = []
    for talent_id in range(starting_id, starting_id + num_of_talents):
        name = fake.name()
        num_skills = random.randint(5, 10)
        skills = weighted_skill_selection(skills_with_weights, num_skills)
        skills_string = ", ".join(skills)
        experience = random.randint(2, 31)
        hourly_rate = random.randint(15, 100)
        english_level = random.randint(1, 5)
        talent = {
            "id": talent_id,
            "name": name,
            "skills": skills_string,
            "experience_years": experience,
            "hourly_rate": hourly_rate,
            "english_level": english_level,
        }
        talents.append(talent)

    return talents


def weighted_skill_selection(skills_dict, num_skills):
    skills = list(skills_dict.keys())
    weights = list(skills_dict.values())
    selected_skills = random.choices(skills, weights=weights, k=num_skills)
    return list(set(selected_skills))  # Remove duplicates


def get_skill_frequencies():
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """WITH
                   job_rates AS (
                     SELECT
                       MIN(jd.salary_from) AS salary_from_min_rate,
                       MAX(jd.salary_from) AS salary_from_max_rate,
                       MIN(jd.salary_to) AS salary_to_min_rate,
                       MAX(jd.salary_to) AS salary_to_max_rate,
                       AVG(salary_from) AS avg_salary_from,
                       AVG(salary_to) AS avg_salary_to
                     FROM job_detail jd  
                     WHERE salary_from > 0
                       AND salary_to > 0
                       AND salary_from < 500
                       AND salary_to < 500
                   ),

                   job_data AS (
                     SELECT DISTINCT
                       rj.job_id,
                       rj.title,
                       CASE
                         WHEN jd.salary_to IS NOT NULL AND jd.salary_to < 500
                             THEN ROUND(jd.salary_to + (SELECT avg_salary_to
                                                          FROM job_rates))
                         WHEN jd.salary_to IS NULL OR jd.salary_to > 500
                             THEN ROUND((SELECT avg_salary_to FROM job_rates))
                       END AS job_rate,
                       jd.parsed_skills
                     FROM raw_jobs rj
                     JOIN job_detail jd 
                       ON rj.job_id = jd.job_id
                     WHERE jd.parsed_skills IS NOT NULL 
                       AND jd.parsed_skills != ''
                       AND jd.parsed_skills != 'skills'
                   ),

                   job_skills AS (
                     SELECT
                       jd.job_id,
                       jd.title,
                       jd.job_rate,
                       TRIM(LOWER(skill)) AS skill
                     FROM job_data jd
                     JOIN LATERAL (
                       SELECT DISTINCT skill
                         FROM UNNEST(
                             STRING_TO_ARRAY(jd.parsed_skills, ',')) AS skill
                     ) AS s(skill) ON TRUE
                   )

               SELECT skill,
                      COUNT(*) AS frequency,
                      ROW_NUMBER() OVER (ORDER BY count(*) DESC) AS rank
                 FROM job_skills
                GROUP BY skill
                ORDER BY frequency DESC;
               """
            )
        )
        skills = [(row[0], row[1], row[2]) for row in result]
    return skills


def categorize_skills(skills_with_freq):
    total_skills = len(skills_with_freq)
    high_demand_cutoff = int(total_skills * 0.2)
    medium_demand_cutoff = int(total_skills * 0.5)
    high_demand_skills = [
        skill for skill, _ in skills_with_freq[:high_demand_cutoff]
    ]
    medium_demand_skills = [
        skill
        for skill, _ in skills_with_freq[
            high_demand_cutoff: medium_demand_cutoff]
    ]
    low_demand_skills = [
        skill for skill, _ in skills_with_freq[medium_demand_cutoff:]
    ]
    return high_demand_skills, medium_demand_skills, low_demand_skills


def get_db_jobs():
    with engine.connect() as conn:
        return pd.DataFrame(
            conn.execute(
                text(
                    """WITH
                job_rates AS (
                  SELECT min(jd.salary_from) AS salary_from_min_rate,
                         max(jd.salary_from) AS salary_from_max_rate,
                         min(jd.salary_to) AS salary_to_min_rate,
                         max(jd.salary_to) AS salary_to_max_rate,
                         avg(salary_from) AS avg_salary_from,
                         avg(salary_to) AS avg_salary_to
                    FROM job_detail jd
                    WHERE salary_from > 0
                    AND salary_to > 0
                    AND salary_from < 500
                    AND salary_to < 500
                )
            SELECT DISTINCT rj.job_id,
                   rj.title,
                   CASE WHEN jd.salary_to IS NOT NULL AND jd.salary_to < 500
                        THEN round((jd.salary_to + (SELECT avg_salary_to
                                                      FROM job_rates)))
                        WHEN jd.salary_to IS NULL OR jd.salary_to > 500
                        THEN round((SELECT avg_salary_to FROM job_rates))
                   END AS job_rate,
                   jd.parsed_skills,
                   FLOOR(RANDOM() * 30) + 2 AS requested_experience
              FROM raw_jobs rj
              JOIN job_detail jd
                ON rj.job_id  = jd.job_id
             WHERE jd.parsed_skills IS NOT NULL
               AND jd.parsed_skills != ''
               AND jd.parsed_skills != 'skills'"""
                )
            ).fetchall()
        )


def main():
    parser = argparse.ArgumentParser(description="Process actions.")
    parser.add_argument("action", type=str, help="Action to be performed")
    args = parser.parse_args()
    action = args.action

    if action == "search-jobs":
        print("Searching Jobs...")
        list_of_skills = [
            "React",
            "Node.js",
            "JavaScript",
            "Java",
            "SQL",
            "Machine Learning",
            "Artificial Intelligence",
            "Python",
            "Golang",
            "Devops",
        ]

        for skill in list_of_skills:
            df = get_jobs(skill)
            insert_raw_jobs(df, "raw_jobs")

    elif action == "update-job-rates":
        print("Updating Job Rates...")
        get_job_salaries()

    elif action == "parse-skills":
        print("Parsing Skills...")
        job_detail = get_job_details()
        retry_count = 0
        for job in job_detail:
            data_dict = extract_skills(job[1], retry_count)
            print(data_dict)
            retry_count = data_dict["rtc"]
            if "skills" in data_dict:
                if data_dict["skills"] != "skills":
                    update_job_skills(job[0], data_dict["skills"])

    elif action == "mock-talents":
        skills_freq = get_skill_frequencies()
        df_skills = pd.DataFrame(skills_freq, columns=["skill", 
                                                       "frequency",
                                                       "rank"])

        total_frequency = df_skills["frequency"].sum()
        print(f"Total Frequency: {total_frequency}")

        df_skills.sort_values("rank", inplace=True)

        # Calculate cumulative frequency
        df_skills["cumulative_frequency"] = df_skills["frequency"].cumsum()

        # Calculate cumulative percentage
        df_skills["cumulative_percentage"] = (
            df_skills["cumulative_frequency"] / total_frequency
        ) * 100

        pprint(df_skills.head(20))

        # Find the rank where cumulative percentage >= 70%
        rank_70 = df_skills[df_skills
                            ["cumulative_percentage"] >= 70].iloc[0]["rank"]

        cumulative_percentage_at_rank_70 = df_skills[
            df_skills["rank"] == rank_70][
            "cumulative_percentage"
        ].values[0]

        print(
            f"""Cumulative percentage reaches
            {cumulative_percentage_at_rank_70:.2f}% at rank {rank_70}"""
        )

        # Find the rank where cumulative percentage >= 90%
        rank_90_data = df_skills[
            df_skills["cumulative_percentage"] >= 90].iloc[0]

        rank_90 = rank_90_data["rank"]

        cumulative_percentage_at_rank_90 = rank_90_data["cumulative_percentage"]
        print(
            f"""Cumulative percentage reaches
            {cumulative_percentage_at_rank_90:.2f}% at rank {rank_90}"""
        )

        plt.figure(figsize=(10, 6))
        plt.plot(df_skills['rank'], df_skills['cumulative_percentage'])
        plt.axhline(y=80, color='r', linestyle='--', label='80% Threshold')
        plt.axvline(x=rank_90, color='g', linestyle='--', label=f'Rank {int(rank_90)}')
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Percentage of Frequency')
        plt.title('Cumulative Frequency Distribution of Skills')
        plt.legend()
        plt.grid(True)
        plt.show()

        # a, b, c = categorize_skills(skills_freq)

        # print(a, b, c)

        # job_skills = get_unique_job_skills()
        # generate_mock_talents(job_skills, 50)

        # Assign demand levels based on ranks
        def assign_demand_level(rank):
            if rank <= rank_70:
                return "High"
            elif rank_70 < rank <= rank_90:
                return "Medium"
            else:
                return "Low"

        df_skills["demand_level"] = df_skills["rank"].apply(
            assign_demand_level)

        skills_with_weights = {}
        for _, row in df_skills.iterrows():
            skill = row["skill"]
            demand_level = row["demand_level"]
            if demand_level == "High":
                weight = 10
            elif demand_level == "Medium":
                weight = 5
            else:
                weight = 1
            skills_with_weights[skill] = weight

        # Talent allocations based on 70/20/10 split
        total_talents = 10000
        high_demand_talents_num = int(0.70 * total_talents)
        medium_demand_talents_num = int(0.20 * total_talents)
        low_demand_talents_num = (
            total_talents - high_demand_talents_num - medium_demand_talents_num
        )

        # Generate talents
        high_demand_talents = generate_weighted_talents(
            num_of_talents=high_demand_talents_num,
            starting_id=1,
            skills_with_weights=skills_with_weights,
        )

        medium_demand_talents = generate_weighted_talents(
            num_of_talents=medium_demand_talents_num,
            starting_id=high_demand_talents_num + 1,
            skills_with_weights=skills_with_weights,
        )

        low_demand_talents = generate_weighted_talents(
            num_of_talents=low_demand_talents_num,
            starting_id=high_demand_talents_num +
            medium_demand_talents_num + 1,
            skills_with_weights=skills_with_weights,
        )

        all_talents = high_demand_talents + medium_demand_talents +\
            low_demand_talents

        df_talents = pd.DataFrame(all_talents)
        #df_talents.to_csv("mock_talents.csv", index=False)

        # Verify the talent distribution
        all_skills_in_talents = []
        for talent in all_talents:
            all_skills_in_talents.extend(talent["skills"])

        talent_skill_counts = Counter(all_skills_in_talents)

        # Create DataFrame for analysis
        df_talent_skills = pd.DataFrame(
            talent_skill_counts.items(), columns=["skill", "count"]
        )

        # Merge with original skill data to get ranks and demand levels
        df_talent_skills = df_talent_skills.merge(
            df_skills[["skill", "rank", "demand_level"]],
            on="skill",
            how="left"
        )

        # Calculate total counts per demand level
        demand_level_counts = df_talent_skills.groupby(
            "demand_level")["count"].sum()

        # Calculate percentages
        total_skills_count = df_talent_skills["count"].sum()
        demand_level_percentages = (
            (demand_level_counts / total_skills_count) * 100)

        print("Skill distribution in generated talents:")
        print(demand_level_percentages)

        # Optionally, plot the distribution
        demand_level_percentages.plot(
            kind="bar", title="Skill Distribution by Demand Level"
        )
        plt.xlabel("Demand Level")
        plt.ylabel("Percentage of Skills")
        plt.show()

    elif action == "get-db-jobs":
        jobs = get_db_jobs()
        jobs.to_csv("jobs.csv")
