import os
from openai import OpenAI
from datetime import datetime, timedelta
import re

# Remove comments to test locally
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_responses(repository_name):
    conversation = []
    responses = []
    total_tokens = 0
    i = 0

    system_content = "You are a Machine Learning Engineer training a Senior Full Stack Software Engineer how to build and deploy scalable, data-intensive, machine learning solutions that leverage the use of the machine learning pipeline which is sourcing, preprocessing, and modeling data, and finally deploying data to production. Respond in Markdown format."

    prompts = [
        f"Analyze the machine learning {repository_name} repository. Pinpoint objectives and benefits to a specific audience. Specific machine learning algorithm. Sourcing, preprocessing, modeling and deploying strategies. Links to all tools and libraries",
        f"Expand and analyze the sourcing data strategy. Could you recommend specific tools or methods that are well-suited for efficiently collecting this data for the project that covers all relevant aspects of the problem domain? Include how these tools integrate within our existing technology stack to streamline the data collection process, ensuring the data is readily accessible and in the correct format for analysis and model training for our project.",
        f"To optimize the development and effectiveness of the project's objectives, we need a detailed analysis of the feature extraction, feature engineering needed for the project's success, aiming to enhance both the interpretability of the data and the performance of the project's machine learning model. Include recommendations for all variable names.",
        f"Could you outline the specific problems that might arise with this data? Furthermore, how can data preprocessing practices be strategically employed to solve these issues, ensuring our data remains robust, reliable, and conducive to high-performing machine learning models? Please provide insights that are directly relevant to the unique demands and characteristics of our project, rather than general data preprocessing benefits.",
        f"Generate the production-ready code required for preprocessing the data",
        f"With the feature extraction, feature engineering, and preprocessing in place, please recommend the metadata management needed for our project's success, please provide insights that are directly relevant to the unique demands and characteristics of our project, rather than general metadata management benefits",
        f"With the foundational aspects of our project now laid out, including the identification of feature engineering, metadata management, selection of data collection tools, and a proactive approach to data preprocessing, we are poised to develop a comprehensive modeling strategy. This strategy must adeptly handle the complexities of the project's objectives and benefits accurately. Could you recommend a modeling strategy that is particularly suited to the unique challenges and data types presented by our project? Additionally, please emphasize the most crucial step within this recommended strategy, explaining why it is particularly vital for the success of our project. This step should address the intricacies of working with our specific types of data and the overarching goal of the project.",
        f"With the modeling strategy for our project now taking shape, our attention turns to the concrete tools and technologies that will bring this vision to life. Given the project's reliance on processing and analyzing data, it's imperative that our toolkit not only aligns with these data types but also integrates seamlessly into our existing workflow. Could you provide specific recommendations for tools and technologies tailored to our project's data modeling needs? For each tool recommended, please include: A brief description of how the tool fits into our modeling strategy, particularly in handling our project's data. Insights into how each tool integrates with our current technologies. Specific features or modules within these tools that will be most beneficial for our project objectives. Links to official documentation or resources that offer in-depth information about the tool and its use cases relevant to our project. This detailed inquiry aims to ensure that our selection of data modeling tools is not only strategic but also pragmatically focused on enhancing our project's efficiency, accuracy, and scalability.",
        f"As we prepare to test our project's model, we need to generate a large, fictitious dataset that mimics real-world data relevant to our project. Generate a python script for creating one. We seek advice on: Methodologies for creating a realistic mocked dataset. Recommended tools for dataset creation and validation, compatible with our tech stack. Strategies for incorporating real-world variability into the data. Structuring the dataset to meet the model's training and validation needs. Resources or frameworks to expedite the creation of this mocked data file, with links to documentation or tutorials if available. Our goal is to ensure the dataset accurately simulates real conditions and integrates seamlessly with our model, enhancing its predictive accuracy and reliability.",
        f"To further refine our approach for our project, we are interested in visualizing an example of the mocked dataset. This example should mimic the real-world data we plan to use, tailored to our project's objectives. Could you provide a sample file that includes: A few rows of data representing what is relevant to our project. An indication of how these data points are structured, including feature names and types. Any specific formatting that will be used for model ingestion, especially concerning its representation for the project. This sample will serve as a visual guide to better understand the mocked data's structure and composition, akin to adding an image for clarity in articles.",
        f"To enhance the project's transition into production, we now seek to develop a production-ready code file for the model(s) utilizing the preprocessed dataset. The focus is on ensuring the code adheres to the high standards of quality, readability, and maintainability observed in large tech companies. Could you provide: A code file or snippet that is structured for immediate deployment in a production environment, specifically designed for our model's data. Detailed comments within the code that explain the logic, purpose, and functionality of key sections, following best practices for documentation. Any conventions or standards for code quality and structure that are commonly adopted in large tech environments, ensuring our codebase remains robust and scalable. This request aims at securing a clear, well-documented, and high-quality code example that can serve as a benchmark for developing our project's production-level machine learning model.",
        f"To effectively deploy the model into production, we require a step-by-step deployment plan, including references to necessary tools. Please provide: A brief outline of the deployment steps for our machine learning model, from pre-deployment checks to live environment integration. Key tools and platforms recommended for each step, with direct links to their official documentation. This guide should empower our team with a clear roadmap and the confidence to execute the deployment independently.",
        f"For the final phase of preparing the project for production, we need to create a Dockerfile that encapsulates our environment and dependencies, tailored specifically to our project's performance needs. Please provide: A production-ready Dockerfile with configurations optimized for handling the objectives of our project. Specific instructions within the Dockerfile that address our project's unique performance and scalability requirements. This Dockerfile should serve as a robust container setup, ensuring optimal performance for our specific use case.",
        f"To fully understand the impact of the {repository_name} project, we need to identify the diverse user groups that will interact with the application. Please provide: A list of the types of users who will benefit from using the application. For each user type, a user story that includes: A brief scenario illustrating their specific pain points. How the application addresses these pain points and the benefits it offers. Which particular file or component of the project facilitates this solution. This information will help showcase the project's wide-ranging benefits and how it serves different audiences, enhancing our understanding of its value proposition.",
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

            print(f"\n\n\nSTART: starting openai call: {conversation}\n\n\n")

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

            print(f"\n\n\nSUCCESS: openai call successful\n\n\n")

            print(f"RESPONSE: {responses[i]}")

            conversation.append({"role": "assistant", "content": responses[i]})

        except Exception as e:
            print(f"ERROR: An error occurred: {str(e)}")

        i += 1

    print(f"\n\n\nTOTAL TOKENS USED: {total_tokens}\n\n\n")

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
    print("START: script start")

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
                f"SUCCESS: Article for '{repository_name}' generated and saved as {markdown_filename}."
            )

            print(f"Write the updated repository_names.txt back to the file")
            with open("repository_names.txt", "w") as file:
                for remaining_repository_names in repository_names:
                    file.write(f"{remaining_repository_names}\n")
            print(f"UPDATED: repository_names.txt updated")
        else:
            print(f"ERROR: Failed to generate article for '{repository_name}'.")
    else:
        print("No more repository_names to generate articles for.")
    print("script end")


if __name__ == "__main__":
    main()
