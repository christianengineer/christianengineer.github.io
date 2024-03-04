import os
from openai import OpenAI
from datetime import datetime, timedelta
import re

# # Remove comments to test locally
# from dotenv import load_dotenv
# # Load environment variables from .env file
# load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_responses(repository_name):
    conversation = []
    responses = []
    total_tokens = 0
    i = 0

    system_content = "You are a Machine Learning Engineer training a Senior Full Stack Software Engineer how to build and deploy scalable, data-intensive, machine learning solutions that leverage the use of the machine learning pipeline which is sourcing, cleansing, and modeling data, and finally deploying data to production. Respond in Markdown format."

    prompts = [
        f"Expand on the machine learning {repository_name} repository. Objectives, sourcing, cleansing, modeling and deploying strategies, with their choosen tools and libraries",
        f"Expand on the MLOps infrastructure",
        f"Generate a scalable folder and file structure for this repository",
        f"Expand on the sourcing directory and its files",
        f"Expand on the cleansing directory and its files",
        f"Expand on the modeling directory and its files",
        f"Expand on the deployment directory and its files",
        f"Generate a large fictitious mocked data structured data file for training the model",
        f"Expand on the modeling strategy step by step and prioritize on the most importan step for this project",
        f"Generate the code required for training the model with the mocked data. Include file path",
        f"Generate a list of type of users that will use the {repository_name} application. Include a user story for each type of user and which file will accomplish this",
    ]

    while i < len(prompts):
        try:
            if i == 0:
                conversation = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompts[i]},
                ]
            else:
                conversation.append({"role": "user", "content": prompts[i]})

            print(f"\n\n\nstarting openai call: {conversation}\n\n\n")

            response = client.chat.completions.create(
                # model="gpt-4-1106-preview", messages=conversation
                # model="gpt-3.5-turbo-1106",
                model="gpt-3.5-turbo-0125",
                messages=conversation,
            )

            if response.choices and response.choices[0].message:
                responses.append(response.choices[0].message.content)
                total_tokens += response.usage.total_tokens
            else:
                raise Exception("No valid response received")

            print(f"\n\n\ni: {i}\n\n\n")

            print(f"\n\n\nopenai call successful\n\n\n")

            print(f"response: {responses[i]}")

            conversation.append({"role": "assistant", "content": responses[i]})

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        i += 1

    print(f"\n\n\n{total_tokens}\n\n\n")

    return responses


def format_title_to_url(title):
    # Truncate the title at the first closing parenthesis ")"
    first_parenthesis_index = title.find(")")
    if first_parenthesis_index != -1:
        title = title[: first_parenthesis_index + 1]

    title = title.lower()
    # Remove special characters
    title = re.sub(r"[^a-z0-9\s-]", "", title)
    # Replace spaces with hyphens
    title = re.sub(r"\s+", "-", title).strip("-")
    return title


def main():
    print("script start")

    results = []

    with open("repository_names.txt", "r") as file:
        repository_names = [line.strip() for line in file if line.strip()]

    if repository_names:
        repository_name = repository_names.pop(0)

        permalink_url = format_title_to_url(repository_name)

        today_date = datetime.now().strftime("%Y-%m-%d")

        markdown_filename = f"_posts/{today_date}-{permalink_url}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                md_file.write(
                    f"---\ntitle: {repository_name}\ndate: {today_date}\npermalink: posts/{permalink_url}\n---\n\n"
                )

        results = generate_responses(repository_name)

        combined_result = "\n\n".join(results)

        if combined_result:
            with open(markdown_filename, "a") as md_file:
                md_file.write(combined_result)

            print(
                f"Article for '{repository_name}' generated and saved as {markdown_filename}."
            )

            print(f"Write the updated repository_names back to the file")
            with open("repository_names.txt", "w") as file:
                for remaining_repository_names in repository_names:
                    file.write(f"{remaining_repository_names}\n")
            print(f"repository_names.txt updated")
        else:
            print(f"Failed to generate article for '{repository_name}'.")
    else:
        print("No more repository_names to generate articles for.")
    print("script end")


if __name__ == "__main__":
    main()
