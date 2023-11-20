import os
from openai import OpenAI
from datetime import datetime, timedelta
import random
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    random_number_of_days = random.randrange(time_between_dates.days)
    return start_date + timedelta(days=random_number_of_days)


def summarize(text):
    # Split the text into sections
    sections = text.split("Summary Section:")
    # Check if a summary section exists
    if len(sections) > 1:
        summary = sections[1].strip()
        return summary
    else:
        return "No summary section found."


def add_message_and_get_response(conversation, user_prompt):
    conversation.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=conversation
    )
    assistant_message = response.choices[0].message.content

    # Summarize the assistant's response before appending
    summarized_response = summarize(
        assistant_message
    )  # Assume 'summarize' is a function you've defined to summarize text
    conversation.append({"role": "assistant", "content": summarized_response})

    return assistant_message


def get_user_prompts_for_article(topic):
    user_prompts = [
        f"In Markdown format, create the technical specifications document of the {topic} repository, focusing on efficient data management and high user traffic handling. Include a description, objectives, and reasons for choosing specific libraries, considering the reader's expert knowledge of each. Conclude with a 'Summary Section' summarizing key points and decisions made in the document.",
        f"Generate a professional scalable file structure for {topic}.",
        f"Design a file detailing the core logic of {topic}. Include file path.",
        f"Create a another file for another core part of {topic}. Include how it integrates with other files.",
        f"Develop another file outlining an additional core logic of {topic}, emphasizing its role in the overall system and interdependencies with previously outlined files.",
        f"Generate list of type of users that will use the {topic} application. Include a user story for each type of user and which file will accomplish this.",
    ]
    return user_prompts


def format_title_to_url(title):
    title = title.lower()
    # Remove special characters
    title = re.sub(r"[^a-z0-9\s-]", "", title)
    # Replace spaces with hyphens
    title = re.sub(r"\s+", "-", title).strip("-")
    return title


def main():
    print("script start")

    results = []

    conversation = [
        {
            "role": "system",
            "content": "You are a large AI Corporation Principal Engineer looking to hire a highly skilled Senior Full Stack Software Engineer",
        }
    ]

    with open("topics.txt", "r") as file:
        topics = [line.strip() for line in file if line.strip()]

    if topics:
        topic = topics.pop(0)

        for prompt in get_user_prompts_for_article(topic):
            response = add_message_and_get_response(conversation, prompt)
            results.append(response)
            print(f"\nprompt successful: {conversation}\n\n{prompt}\n\n{response}\n\n")

        # Define start and end dates
        start_date = datetime(2023, 11, 18)
        end_date = datetime(2023, 11, 19)

        random_date = get_random_date(start_date, end_date).strftime("%Y-%m-%d")

        permalink_url = format_title_to_url(topic)

        # today_date = datetime.now().strftime("%Y-%m-%d")
        markdown_filename = f"_posts/{random_date}-{permalink_url}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                md_file.write(
                    f"---\ntitle: {topic}\ndate: {random_date}\npermalink: posts/{permalink_url}\n---\n\n"
                )

        combined_result = "\n\n".join(results)

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
