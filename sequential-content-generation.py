import os
from openai import OpenAI
from datetime import datetime, timedelta
import random


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    random_number_of_days = random.randrange(time_between_dates.days)
    return start_date + timedelta(days=random_number_of_days)


def add_message_and_get_response(conversation, user_prompt):
    conversation.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(model="gpt-4", messages=conversation)
    assistant_message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def get_user_prompts(topic):
    user_prompts = [
        f"In Markdown format, expand on the {topic} repository. Description, goals and libraries to be used for efficient data handling and scalable user traffic.",
        f"In Markdown format, generate a scalable file structure for the {topic} repository.",
        f"In Markdown format, generate a fictitious file for {topic}. This file will handle the logic for {topic}. Include the folder location.",
        # main_title
        f"Generate a succinct and engaging title for a comprehensive development plan aimed at creating a scalable {topic} for a large AI Corporation. The title should encapsulate the essence of a project that encompasses design, development, deployment, and data handling strategies. It should reflect the project’s commitment to handling large volumes of data, robust scalability testing, and the integration of cloud technologies and AI for superior performance for high user traffic. Consider a title that conveys innovation, technological advancement, and scalability.",
        # seo_title
        f"Develop a concise, SEO-focused file name tailored for web URLs, derived from an in-depth article about {topic}. The file name should integrate targeted SEO keywords to maximize search engine visibility, appealing to a technically adept audience. Ensure it's both engaging and directly related to the article's subject. Use a simple format with only letters and hyphens to replace spaces, maintaining web URL standards. Do not include the '.md' extension. The file name should effectively represent the article's core theme and be optimized for high-ranking search results.",
    ]
    return user_prompts


def main():
    print("script start")

    results = []

    conversation = [
        {
            "role": "system",
            "content": "You are a large AI Corporation looking to hire a highly skilled Full Stack Software Engineer",
        }
    ]

    with open("topics.txt", "r") as file:
        topics = [line.strip() for line in file if line.strip()]

    if topics:
        topic = topics.pop(0)

        for prompt in get_user_prompts(topic):
            response = add_message_and_get_response(conversation, prompt)
            results.append(response)
            print(f"\nprompt successfull: {prompt}\n{response}\n\n")

        # Define start and end dates
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 11, 1)

        random_date = get_random_date(start_date, end_date).strftime("%Y-%m-%d")

        permalink_value = results[4].replace('"', "")

        main_title = results[3]

        # today_date = datetime.now().strftime("%Y-%m-%d")
        markdown_filename = f"_posts/{random_date}-{permalink_value}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                md_file.write(
                    f"---\ntitle: {main_title}\ndate: {random_date}\npermalink: posts/{permalink_value}\n---\n\n"
                )

        combined_result = "\n\n".join(results[:3])

        if combined_result:
            with open(markdown_filename, "a") as md_file:
                md_file.write(combined_result)

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
