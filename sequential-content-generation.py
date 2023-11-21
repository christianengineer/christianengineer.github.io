import os
from openai import OpenAI
from datetime import datetime, timedelta
import random
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_responses(repository_name):
    responses = []
    i = 0

    system_content = "You are a AI Startup Founder seeking a Senior Full Stack Software Engineer skilled in creating scalable AI applications. Preference for candidates with impressive Open Source project experience, demonstrating expertise in developing large-scale AI Applications. Respond in Markdown format."

    prompts = [
        f"Expand on the {repository_name} repository. Description, Objectives and libraries used.",
    ]

    while i < len(prompts):
        try:
            if i == 0 or i == 1:
                conversation = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompts[i]},
                ]
            else:
                if len(responses) > 3 and responses[3].strip():
                    assistant_content = f"{responses[1]} {responses[3]}"
                else:
                    assistant_content = responses[1] if len(responses) > 1 else ""

                conversation = [
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": prompts[i]},
                ]

            print(f"\n\n\nstarting openai call: {conversation}\n\n\n")

            response = client.chat.completions.create(
                model="gpt-4-1106-preview", messages=conversation
            )

            if response.choices and response.choices[0].message:
                responses.append(response.choices[0].message.content)
            else:
                raise Exception("No valid response received")

            print(f"\n\n\ni: {i}\n\n\n")

            print(f"\n\n\nopenai call successful\n\n\n")

            print(f"response: {responses[i]}")

            if i == 0:
                prompts.extend(
                    [
                        f"Summerize the list of key components, scalable strategies, and high traffic features for the {repository_name} application: \n{responses[0]}",
                        f"Generate a scalable file structure for the {repository_name} repository.",
                    ]
                )

            if i == 2:
                prompts.extend(
                    [
                        f"Summarize the key components and the scalable organizational approach of the file structure: \n{responses[2]}",
                        f"Generate a fictitious file for one of the key component's logic of {repository_name}. Include file path.",
                        f"Generate a fictitious file for one of the key component's for the AI logic of the {repository_name} repository. Include file path.",
                        f"Generate a fictitious file for one of the key component's for core logic of the {repository_name} repository. Include file path.",
                        f"Generate a fictitious file for one of the key component's for the API logic of the {repository_name} repository. Include file path.",
                        f"Generate a list of type of users that will use the {repository_name} application. Include a user story for each type of user and which file will accomplish this."
                        # "prompt 0",
                        # f"generate a summary of {response[0]}",
                        # "prompt 2",
                        # f"generate a summary of {response[2]}",
                        # "prompt 4",
                        # "prompt 5",
                        # "prompt 6",
                        # "prompt 7",
                    ]
                )

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        i += 1

    return responses


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
