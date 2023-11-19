import os
from openai import OpenAI
from datetime import datetime, timedelta
import random

# Initialize the OpenAI client
client = OpenAI(
    # api_key=os.getenv("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY")  # Get the API key from environment variable
)


def get_random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    random_number_of_days = random.randrange(time_between_dates.days)
    return start_date + timedelta(days=random_number_of_days)


def generate_main_title(topic):
    print("generate_main_title start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Generate a succinct and engaging title for a comprehensive development plan aimed at creating a scalable {topic} for a large AI Corporation. The title should encapsulate the essence of a project that encompasses design, development, deployment, and data handling strategies, along with machine learning model training and API integration. It should reflect the project’s commitment to handling large volumes of data, robust scalability testing, and the integration of cloud technologies and AI for superior performance and user experience. Consider a title that conveys innovation, technological advancement, and scalability.",
            }
        ],
        model="gpt-4",
    )
    print("generate_main_title end")
    return response.choices[0].message.content.strip()


def generate_seo_url_title(topic):
    print("generate_seo_url_title start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Develop a concise, SEO-focused file name tailored for web URLs, derived from an in-depth article about {topic}. The file name should integrate targeted SEO keywords to maximize search engine visibility, appealing to a technically adept audience. Ensure it's both engaging and directly related to the article's subject. Use a simple format with only letters and hyphens to replace spaces, maintaining web URL standards. Do not include the '.md' extension. The file name should effectively represent the article's core theme and be optimized for high-ranking search results.",
            }
        ],
        model="gpt-4",
    )
    print("generate_seo_url_title end")
    return response.choices[0].message.content.strip()


def generate_article(topic):
    print("generate_article start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                # "content": f"Develop a detailed plan in Markdown format for {topic}, focusing on a scalable file structure and collaborative environment for thousands of engineers. Address the architectural design, emphasizing modular, reusable components and a microservices approach. Include a strategy for efficient data handling, robust machine learning model training, and seamless API integration. Tackle the challenges of high data volumes and concurrent user activities. Incorporate code snippets demonstrating key scalability solutions. Plan for phased rollouts, scalability testing, continuous integration, and deployment. Emphasize cloud and AI technologies to enhance performance and user experience, ensuring a flexible, scalable, and maintainable system.",
                "content": f"Create an advanced development blueprint in Markdown format for {topic} aimed at large-scale AI corporations, designed to captivate and motivate highly skilled software engineers. Focus on presenting a well-structured, modular design, and development strategy, incorporating concise, clear code snippets that offer quick, high-level insights. These snippets should demonstrate innovative solutions for scalable data handling, efficient machine learning model training, and robust API integration. Address the complexities of managing vast data and high user traffic. Include a strategic phased rollout with rigorous scalability testing. Emphasize the utilization of cloud technologies and AI to optimize performance and user experience, crafting a narrative that excites and encourages immediate engagement from top-tier developers.",
            }
        ],
        model="gpt-4",
    )
    article = response.choices[0].message.content.strip()
    print("generate_article end")
    return article


def main():
    print("script start")

    with open("topics.txt", "r") as file:
        topics = [line.strip() for line in file if line.strip()]

    if topics:
        topic = topics.pop(0)

        main_title = generate_main_title(topic)
        print(f"main_title: {main_title}")

        seo_title = generate_seo_url_title(main_title)
        print(f"seo_title: {seo_title}")

        # Define start and end dates
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 11, 1)

        random_date = get_random_date(start_date, end_date).strftime("%Y-%m-%d")

        # today_date = datetime.now().strftime("%Y-%m-%d")
        markdown_filename = f"_posts/{random_date}-{seo_title}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                md_file.write(
                    f"---\ntitle: {main_title}\ndate: {random_date}\npermalink: posts/{seo_title}\n---\n\n"
                )

        article = generate_article(main_title)

        if article:
            with open(markdown_filename, "a") as md_file:
                md_file.write(article)

            print(f"Article for '{topic}' generated and saved as {markdown_filename}.")

            print(f"Write the updated topics back to the file")
            with open("topics.txt", "w") as file:
                for remaining_topic in topics:
                    file.write(f"{remaining_topic}\n")
            print(f"topics.txt updated")
        else:
            print(f"Failed to generate article for '{topic}'.")
    else:
        print("No more topics to generate articles for.")
    print("script end")


if __name__ == "__main__":
    main()
