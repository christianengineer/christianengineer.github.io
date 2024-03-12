import os
from openai import OpenAI
from datetime import datetime, timedelta
import re

# # Remove comments to test locally
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_post(project_name):
    conversation = []
    responses = []
    total_tokens = 0
    i = 0

    # espanol
    # system_content = f"Eres un Ingeniero de Aprendizaje Automático documentando cómo preparar, construir y desplegar una solución de aprendizaje automático escalable y lista para producción para: {project_name}. Prioriza resolver el punto de dolor de la audiencia. Responde en formato Markdown."

    # prompts = [
    #     "Identifique objetivos y beneficios para una audiencia específica. Algoritmo de aprendizaje automático específico. Estrategias de obtención, preprocesamiento, modelado y despliegue. Enlaces a todas las herramientas y bibliotecas.",
    #     "Expanda y analice la estrategia de obtención de datos. ¿Podría recomendar herramientas o métodos específicos que sean adecuados para recopilar eficientemente estos datos para el proyecto que cubre todos los aspectos relevantes del dominio del problema? Incluya cómo estas herramientas se integran dentro de nuestro actual stack tecnológico para agilizar el proceso de recolección de datos, asegurando que los datos estén fácilmente accesibles y en el formato correcto para el análisis y entrenamiento del modelo para nuestro proyecto.",
    #     "Para optimizar el desarrollo y la efectividad de los objetivos del proyecto, es crucial realizar un análisis detallado de la extracción y la ingeniería de características necesarias para el éxito del proyecto, buscando mejorar tanto la interpretabilidad de los datos como el rendimiento del modelo de aprendizaje automático del proyecto. Incluya recomendaciones para todos los nombres de variables.",
    #     "Con la extracción y la ingeniería de características, y el preprocesamiento en su lugar, por favor recomiende la gestión de metadatos necesaria para el éxito de nuestro proyecto, proporcionando ideas que sean directamente relevantes para las demandas y características únicas de nuestro proyecto, en lugar de los beneficios generales de la gestión de metadatos.",
    #     "¿Podría delinear los problemas específicos que podrían surgir con los datos de nuestro proyecto? Además, ¿cómo pueden las prácticas de preprocesamiento de datos emplearse estratégicamente para resolver estos problemas, asegurando que nuestros datos permanezcan robustos, confiables y conducentes a modelos de aprendizaje automático de alto rendimiento? Proporcione ideas que sean directamente relevantes para las demandas y características únicas de nuestro proyecto, en lugar de los beneficios generales del preprocesamiento de datos.",
    #     "Para avanzar en nuestro proyecto, necesitamos preprocesar nuestros datos para asegurarnos de que estén listos para el entrenamiento del modelo. Proporcione: Un archivo de código que describa los pasos de preprocesamiento necesarios adaptados a nuestra estrategia de preprocesamiento. Comentarios dentro del código explicando cada paso de preprocesamiento y su importancia para las necesidades específicas de nuestro proyecto. Este código será fundamental en la preparación de nuestros datos para un entrenamiento y análisis de modelo efectivos.",
    #     "Con los aspectos fundamentales de nuestro proyecto ya definidos, incluida la identificación de la ingeniería de características, la gestión de metadatos, la selección de herramientas de recolección de datos y un enfoque proactivo para el preprocesamiento de datos, estamos preparados para desarrollar una estrategia de modelado integral. Esta estrategia debe manejar con destreza las complejidades de los objetivos y beneficios del proyecto de manera precisa. ¿Podría recomendar una estrategia de modelado que sea especialmente adecuada para los desafíos únicos y tipos de datos presentados por nuestro proyecto? Además, por favor, enfatice el paso más crucial dentro de esta estrategia recomendada, explicando por qué es particularmente vital para el éxito de nuestro proyecto. Este paso debe abordar las complejidades de trabajar con nuestros tipos específicos de datos y el objetivo general del proyecto.",
    #     "Con la estrategia de modelado para nuestro proyecto tomando forma, nuestra atención se dirige a las herramientas y tecnologías concretas que darán vida a esta visión. Dada la dependencia del proyecto en el procesamiento y análisis de datos, es imperativo que nuestro conjunto de herramientas no solo se alinee con estos tipos de datos sino que también se integre sin problemas en nuestro flujo de trabajo existente. ¿Podría proporcionar recomendaciones específicas de herramientas y tecnologías adaptadas a las necesidades de modelado de datos de nuestro proyecto? Para cada herramienta recomendada, incluya: Una breve descripción de cómo la herramienta se ajusta a nuestra estrategia de modelado, particularmente en el manejo de los datos de nuestro proyecto y la solución al punto de dolor. Perspectivas sobre cómo cada herramienta se integra con nuestras tecnologías actuales. Características o módulos específicos dentro de estas herramientas que serán más beneficiosos para los objetivos de nuestro proyecto. Enlaces a documentación oficial o recursos que ofrecen información detallada sobre la herramienta y sus casos de uso relevantes para nuestro proyecto. Esta consulta detallada tiene como objetivo asegurar que nuestra selección de herramientas de modelado de datos no solo sea estratégica sino también prácticamente enfocada en mejorar la eficiencia, precisión y escalabilidad de nuestro proyecto.",
    #     "Mientras nos preparamos para probar el modelo de nuestro proyecto, necesitamos generar un gran conjunto de datos ficticios que imite los datos del mundo real relevantes para nuestro proyecto que utiliza nuestras estrategias de extracción de características, ingeniería de características y gestión de metadatos. Genere un script en Python para crear este conjunto de datos e incluya todos los atributos de las características necesarias para nuestro proyecto. El script debe tener herramientas recomendadas para la creación y validación del conjunto de datos, compatibles con nuestro stack tecnológico, una estrategia para incorporar variabilidad del mundo real y satisfacer las necesidades de entrenamiento y validación del modelo de nuestro proyecto, asegurando que el conjunto de datos simule con precisión las condiciones reales e integre a la perfección con nuestro modelo, mejorando su precisión predictiva y fiabilidad.",
    #     "Para refinar aún más nuestro enfoque para nuestro proyecto, estamos interesados en visualizar un ejemplo del conjunto de datos simulados. Este ejemplo debe imitar los datos del mundo real que planeamos usar, adaptados a los objetivos de nuestro proyecto. ¿Podría proporcionar un archivo de muestra que incluya: Unas pocas filas de datos que representen lo que es relevante para nuestro proyecto? Una indicación de cómo se estructuran estos puntos de datos, incluyendo nombres y tipos de características. Cualquier formato específico que se utilizará para la ingestión del modelo, especialmente en lo que respecta a su representación para el proyecto. Esta muestra servirá como una guía visual para comprender mejor la estructura y composición de los datos simulados, similar a agregar una imagen para claridad en los artículos.",
    #     "Para mejorar la transición del proyecto a producción, ahora buscamos desarrollar un archivo de código listo para producción para el(los) modelo(s) utilizando el conjunto de datos preprocesado. El enfoque está en asegurar que el código cumpla con los altos estándares de calidad, legibilidad y mantenibilidad observados en las grandes empresas tecnológicas. ¿Podría proporcionar: Un archivo de código o fragmento que esté estructurado para el despliegue inmediato en un entorno de producción, específicamente diseñado para los datos de nuestro modelo? Comentarios detallados dentro del código que expliquen la lógica, propósito y funcionalidad de secciones clave, siguiendo las mejores prácticas para la documentación. ¿Cualquier convención o estándar para la calidad y estructura del código que se adopta comúnmente en entornos tecnológicos grandes, asegurando que nuestra base de código permanezca robusta y escalable? Esta solicitud tiene como objetivo asegurar un ejemplo de código claro, bien documentado y de alta calidad que pueda servir como referencia para desarrollar nuestro modelo de aprendizaje automático a nivel de producción del proyecto.",
    #     "Para desplegar eficazmente el modelo en producción, necesitamos un plan de despliegue paso a paso que sea directamente relevante para las demandas y características únicas de nuestro proyecto, en lugar de un plan de despliegue general, incluyendo referencias a herramientas necesarias. Proporcione: Un esquema breve de los pasos de despliegue para nuestro modelo de aprendizaje automático, desde los chequeos previos al despliegue hasta la integración en el entorno en vivo. Herramientas clave y plataformas recomendadas para cada paso, con enlaces directos a su documentación oficial. Esta guía debería empoderar a nuestro equipo con una hoja de ruta clara y la confianza para ejecutar el despliegue de forma independiente.",
    #     "Para la fase final de preparar el proyecto para producción, necesitamos crear un Dockerfile que encapsule nuestro entorno y dependencias, adaptado específicamente a las necesidades de rendimiento de nuestro proyecto. Proporcione: Un Dockerfile listo para producción con configuraciones optimizadas para manejar los objetivos de nuestro proyecto. Instrucciones específicas dentro del Dockerfile que aborden los requisitos de rendimiento y escalabilidad únicos de nuestro proyecto. Este Dockerfile debería servir como una configuración de contenedor robusta, asegurando un rendimiento óptimo para nuestro caso de uso específico.",
    #     f"Para comprender completamente el impacto del proyecto: {project_name}, necesitamos identificar los diversos grupos de usuarios que interactuarán con la aplicación. Proporcione: Una lista de los tipos de usuarios que se beneficiarán de usar la aplicación. Para cada tipo de usuario, una historia de usuario que incluya: Un escenario breve que ilustre sus puntos de dolor específicos. Cómo la aplicación aborda estos puntos de dolor y los beneficios que ofrece. Qué archivo o componente particular del proyecto facilita esta solución. Esta información ayudará a mostrar los beneficios amplios del proyecto y cómo sirve a diferentes audiencias, mejorando nuestra comprensión de su propuesta de valor.",
    # ]

    # system_content = f"You are a Machine Learning Engineer writing in-depth documentation on how to plan, build, and deploy a scalable, production-ready, machine learning solution for: {project_name}. Exploring the untapped potential of integrating artificial intelligence with this indutry, solving the audience's pain point. Discuss how this integration can revolutionize this industry by enhancing creativity, personalization, and efficiency. Provide real-world examples or hypothetical scenarios illustrating the benefits and challenges of this fusion. Conclude with predictions for the future and a call to action for this industry's stakeholders to explore this innovative synergy for this project. Respond in SEO optimized Markdown format. Do not include a level 1 heading markdown tag."

    # prompts = [
    #     f"Pinpoint objectives and benefits to a specific audience. Specific machine learning algorithm. Sourcing, preprocessing, modeling and deploying strategies. Links to all tools and libraries",
    #     f"Expand and analyze the sourcing data strategy. Could you recommend specific tools or methods that are well-suited for efficiently collecting this data for the project that covers all relevant aspects of the problem domain? Include how these tools integrate within our existing technology stack to streamline the data collection process, ensuring the data is readily accessible and in the correct format for analysis and model training for our project.",
    #     f"To optimize the development and effectiveness of the project's objectives, we need a detailed analysis of the feature extraction, feature engineering needed for the project's success, aiming to enhance both the interpretability of the data and the performance of the project's machine learning model. Include recommendations for all variable names.",
    #     f"With the feature extraction, feature engineering, and preprocessing in place, please recommend the metadata management needed for our project's success, please provide insights that are directly relevant to the unique demands and characteristics of our project, rather than general metadata management benefits",
    #     f"Could you outline the specific problems that might arise with our project's data? Furthermore, how can data preprocessing practices be strategically employed to solve these issues, ensuring our data remains robust, reliable, and conducive to high-performing machine learning models? Please provide insights that are directly relevant to the unique demands and characteristics of our project, rather than general data preprocessing benefits.",
    #     f"To advance our project, we need to preprocess our data to ensure it's ready for model training. Please provide: A code file that outlines the necessary preprocessing steps tailored to our preprocessing strategy. Comments within the code explaining each preprocessing step and its importance to our specific project needs. This code will be pivotal in preparing our data for effective model training and analysis.",
    #     f"With the foundational aspects of our project now laid out, including the identification of feature engineering, metadata management, selection of data collection tools, and a proactive approach to data preprocessing, we are poised to develop a comprehensive modeling strategy. This strategy must adeptly handle the complexities of the project's objectives and benefits accurately. Could you recommend a modeling strategy that is particularly suited to the unique challenges and data types presented by our project? Additionally, please emphasize the most crucial step within this recommended strategy, explaining why it is particularly vital for the success of our project. This step should address the intricacies of working with our specific types of data and the overarching goal of the project.",
    #     f"With the modeling strategy for our project now taking shape, our attention turns to the concrete tools and technologies that will bring this vision to life. Given the project's reliance on processing and analyzing data, it's imperative that our toolkit not only aligns with these data types but also integrates seamlessly into our existing workflow. Could you provide specific recommendations for tools and technologies tailored to our project's data modeling needs? For each tool recommended, please include: A brief description of how the tool fits into our modeling strategy, particularly in handling our project's data and solution to the pain point. Insights into how each tool integrates with our current technologies. Specific features or modules within these tools that will be most beneficial for our project objectives. Links to official documentation or resources that offer in-depth information about the tool and its use cases relevant to our project. This detailed inquiry aims to ensure that our selection of data modeling tools is not only strategic but also pragmatically focused on enhancing our project's efficiency, accuracy, and scalability.",
    #     f"As we prepare to test our project's model, we need to generate a large fictitious dataset that mimics real-world data relevant to our project that uses our feature extraction, feature engineering, and metadata management strategies. Generate a python script for creating this dataset and include all attributes from the features needed for our project. The script needs to have recommended tools for dataset creation and validation, compatible with our tech stack, a strategy for incorporating real-world variability and meet our project's model training and validation needs, ensuring the dataset accurately simulates real conditions and integrates seamlessly with our model, enhancing its predictive accuracy and reliability.",
    #     f"To further refine our approach for our project, we are interested in visualizing an example of the mocked dataset. This example should mimic the real-world data we plan to use, tailored to our project's objectives. Could you provide a sample file that includes: A few rows of data representing what is relevant to our project. An indication of how these data points are structured, including feature names and types. Any specific formatting that will be used for model ingestion, especially concerning its representation for the project. This sample will serve as a visual guide to better understand the mocked data's structure and composition, akin to adding an image for clarity in articles.",
    #     f"To enhance the project's transition into production, we now seek to develop a production-ready code file for the model(s) utilizing the preprocessed dataset. The focus is on ensuring the code adheres to the high standards of quality, readability, and maintainability observed in large tech companies. Could you provide: A code file or snippet that is structured for immediate deployment in a production environment, specifically designed for our model's data. Detailed comments within the code that explain the logic, purpose, and functionality of key sections, following best practices for documentation. Any conventions or standards for code quality and structure that are commonly adopted in large tech environments, ensuring our codebase remains robust and scalable. This request aims at securing a clear, well-documented, and high-quality code example that can serve as a benchmark for developing our project's production-level machine learning model.",
    #     f"To effectively deploy the model into production, we require a step-by-step deployment plan that is directly relevant to the unique demands and characteristics of our project, rather than a general deployment plan, including references to necessary tools. Please provide: A brief outline of the deployment steps for our machine learning model, from pre-deployment checks to live environment integration. Key tools and platforms recommended for each step, with direct links to their official documentation. This guide should empower our team with a clear roadmap and the confidence to execute the deployment independently.",
    #     f"For the final phase of preparing the project for production, we need to create a Dockerfile that encapsulates our environment and dependencies, tailored specifically to our project's performance needs. Please provide: A production-ready Dockerfile with configurations optimized for handling the objectives of our project. Specific instructions within the Dockerfile that address our project's unique performance and scalability requirements. This Dockerfile should serve as a robust container setup, ensuring optimal performance for our specific use case.",
    #     f"To fully understand the impact of the project: {project_name}, we need to identify the diverse user groups that will interact with the application. Please provide: A list of the types of users who will benefit from using the application. For each user type, a user story that includes: A brief scenario illustrating their specific pain points. How the application addresses these pain points and the benefits it offers. Which particular file or component of the project facilitates this solution. This information will help showcase the project's wide-ranging benefits and how it serves different audiences, enhancing our understanding of its value proposition.",
    # ]

    # v3.0 - March 11, 2024
    system_content = f"You are an Artificial Intelligence Engineer writing in-depth documentation to explore, discuss and predict a solution for this project: '{project_name}'. Consider the potential for SEO-optimized content that showcases our expertise and findings, appealing to both search engines and the wider artificial intelligence community. Respond markdown format."

    prompts = [
        f"Based on the project title and objectives, what should be the primary focus be? What would be a target variable name for our model that encapsulates the project's goal? Explain why this target variable is important for this project. Explore example values and how a user could make decisions from these values",
        f"Using fictitious mocked data with a clear description of each input, generate the python script that will create and train a model, and finally print the predicted values of the target variable for the test set and evaluate the model's performance using an appropriate metric",
        f"What secondary target variable, which we'll refer to as '[Secondary_Target_Variable_Name]', could play a critical role in enhancing our predictive model's accuracy and insights? How does '[Secondary_Target_Variable_Name]' complement '[Primary_Target_Variable_Name]' in our quest to achieve groundbreaking results in [specific domain]? Explore example values and how a user could make decisions from these values",
        f"Using fictitious mocked data, generate the python script that will create and train a model using the '[Primary_Target_Variable_Name]' and '[Secondary_Target_Variable_Name]', and finally print the predicted values of the target variable for the test set and evaluate the model's performance using an appropriate metric",
        f"What third target variable, which we'll refer to as '[Third_Target_Variable_Name]', could play a critical role in enhancing our predictive model's accuracy and insights? How does '[Third_Target_Variable_Name]' complement '[Primary_Target_Variable_Name]' and '[Secondary_Target_Variable_Name]' in our quest to achieve groundbreaking results in [specific domain]? Explore example values and how a user could make decisions from these values",
        f"Using fictitious mocked data, generate the python script that will create and train a model using the '[Primary_Target_Variable_Name]', '[Secondary_Target_Variable_Name]' and '[Third_Target_Variable_Name]', and finally print the predicted values of the target variable for the test set and evaluate the model's performance using an appropriate metric",
        f"To fully understand the impact of the project: {project_name}, we need to identify the diverse user groups that will interact with the application. Please provide: A list of the types of users who will benefit from seeing the value of one of the target variables. For each user type, a user story that includes: A brief scenario illustrating their specific pain points. How the value of one of the target variables addresses one of these pain points and the benefits it offers. Which particular component of the project facilitates this solution. This information will help showcase the project's wide-ranging benefits and how it serves different audiences, enhancing our understanding of its value proposition.",
        f"Imagine [User_Name], a [User_Group] facing a specific challenge: [User_Challenge]. Despite their best efforts, [User_Name] encounters [Pain_Point], affecting their [Negative_Impact]. Enter the solution: a project leveraging machine learning to address this challenge, focusing on a key target variable named '[Target_Variable_Name].' This variable holds the potential to transform [User_Name]'s situation by offering [Solution_Feature], designed to [Solution_Benefit]. One day, [User_Name] decides to test this solution. As they engage with the system, they're presented with a [Target_Variable_Value], which is a direct result of the project's machine learning analysis. This value suggests [Specific_Action or Recommendation] that promises to alleviate their [Pain_Point]. Through the narrative, describe how [User_Name] initially reacts to the [Target_Variable_Value] and their decision-making process. Illustrate the moment they follow the [Specific_Action or Recommendation], leading to an outcome where they experience [Positive_Impacts], such as [List_Possible_Benefits]. Conclude the story by reflecting on how the insights derived from '[Target_Variable_Name]'—a mere data point—empowered [User_Name] to make a decision that significantly improved their [Aspect_of_Life or Work]. Highlight the broader implications of this project, showcasing how machine learning can provide actionable insights and real-world solutions to individuals facing [Similar_Challenges]. Emphasize the transformative power of understanding and applying data-driven decisions, thanks to the innovative use of machine learning in [Project_Domain].",
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
                # expensive
                # model="gpt-4-1106-preview",
                # cheap
                # model="gpt-3.5-turbo-1106",
                # cheaper, better
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


def convert_title_to_url_friendly_format(title):
    # Convert title to lowercase first to simplify processing
    title = title.lower()

    # Find the index of the first " - " and truncate the title up to but not including " - "
    first_hyphen_space_index = title.find(" - ")
    if first_hyphen_space_index != -1:
        title = title[:first_hyphen_space_index]

    # Replace hyphens with spaces around them with a single space (though they should no longer exist after truncation)
    title_without_loose_hyphens = re.sub(r"\s+-\s+", " ", title)

    # Remove special characters, allowing only alphanumeric characters, spaces, and hyphens
    title_without_special_chars = re.sub(
        r"[^a-z0-9\s-]", "", title_without_loose_hyphens
    )

    # Replace one or more spaces with a single hyphen and trim trailing hyphens
    url_friendly_title = re.sub(r"\s+", "-", title_without_special_chars).strip("-")

    return url_friendly_title


def main():
    print("START: script start")

    results = []

    with open("assets/projects/project_list_user_problem_solution.txt", "r") as file:
        project_names = [line.strip() for line in file if line.strip()]

    if project_names:
        project_name = project_names.pop(0)

        permalink_url = convert_title_to_url_friendly_format(project_name)

        today_date = datetime.now().strftime("%Y-%m-%d")

        markdown_filename = f"_posts/{today_date}-{permalink_url}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                md_file.write(
                    f"---\ntitle: {project_name}\ndate: {today_date}\npermalink: posts/{permalink_url}\nlayout: article\n---\n\n"
                )

        results = generate_post(project_name)

        combined_result = "\n\n".join(results)

        if combined_result:
            with open(markdown_filename, "a") as md_file:
                md_file.write(combined_result)

            print(
                f"SUCCESS: Article for '{project_name}' generated and saved as {markdown_filename}."
            )

            print(
                f"Write the updated assets/projects/project_list_user_problem_solution.txt back to the file"
            )
            with open(
                "assets/projects/project_list_user_problem_solution.txt", "w"
            ) as file:
                for remaining_project_names in project_names:
                    file.write(f"{remaining_project_names}\n")
            print(
                f"UPDATED: assets/projects/project_list_user_problem_solution.txt updated"
            )
        else:
            print(f"ERROR: Failed to generate article for '{project_name}'.")
    else:
        print("No more projects to generate post for.")
    print("script end")


if __name__ == "__main__":
    main()
