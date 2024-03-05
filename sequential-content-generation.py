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

    # espanol
    # system_content = f"Eres un Ingeniero de Aprendizaje Automático documentando cómo preparar, construir y desplegar una solución de aprendizaje automático escalable y lista para producción para: {repository_name}. Prioriza resolver el punto de dolor de la audiencia. Responde en formato Markdown."

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
    #     f"Para comprender completamente el impacto del proyecto: {repository_name}, necesitamos identificar los diversos grupos de usuarios que interactuarán con la aplicación. Proporcione: Una lista de los tipos de usuarios que se beneficiarán de usar la aplicación. Para cada tipo de usuario, una historia de usuario que incluya: Un escenario breve que ilustre sus puntos de dolor específicos. Cómo la aplicación aborda estos puntos de dolor y los beneficios que ofrece. Qué archivo o componente particular del proyecto facilita esta solución. Esta información ayudará a mostrar los beneficios amplios del proyecto y cómo sirve a diferentes audiencias, mejorando nuestra comprensión de su propuesta de valor.",
    # ]

    system_content = f"You are a Machine Learning Engineer documenting how to prepare, build, and deploy a scalable, production-ready, machine learning solution for: {repository_name}. Prioritize to solve the audience's pain point. Respond in Markdown format."

    prompts = [
        f"Pinpoint objectives and benefits to a specific audience. Specific machine learning algorithm. Sourcing, preprocessing, modeling and deploying strategies. Links to all tools and libraries",
        f"Expand and analyze the sourcing data strategy. Could you recommend specific tools or methods that are well-suited for efficiently collecting this data for the project that covers all relevant aspects of the problem domain? Include how these tools integrate within our existing technology stack to streamline the data collection process, ensuring the data is readily accessible and in the correct format for analysis and model training for our project.",
        f"To optimize the development and effectiveness of the project's objectives, we need a detailed analysis of the feature extraction, feature engineering needed for the project's success, aiming to enhance both the interpretability of the data and the performance of the project's machine learning model. Include recommendations for all variable names.",
        f"With the feature extraction, feature engineering, and preprocessing in place, please recommend the metadata management needed for our project's success, please provide insights that are directly relevant to the unique demands and characteristics of our project, rather than general metadata management benefits",
        f"Could you outline the specific problems that might arise with our project's data? Furthermore, how can data preprocessing practices be strategically employed to solve these issues, ensuring our data remains robust, reliable, and conducive to high-performing machine learning models? Please provide insights that are directly relevant to the unique demands and characteristics of our project, rather than general data preprocessing benefits.",
        f"To advance our project, we need to preprocess our data to ensure it's ready for model training. Please provide: A code file that outlines the necessary preprocessing steps tailored to our preprocessing strategy. Comments within the code explaining each preprocessing step and its importance to our specific project needs. This code will be pivotal in preparing our data for effective model training and analysis.",
        f"With the foundational aspects of our project now laid out, including the identification of feature engineering, metadata management, selection of data collection tools, and a proactive approach to data preprocessing, we are poised to develop a comprehensive modeling strategy. This strategy must adeptly handle the complexities of the project's objectives and benefits accurately. Could you recommend a modeling strategy that is particularly suited to the unique challenges and data types presented by our project? Additionally, please emphasize the most crucial step within this recommended strategy, explaining why it is particularly vital for the success of our project. This step should address the intricacies of working with our specific types of data and the overarching goal of the project.",
        f"With the modeling strategy for our project now taking shape, our attention turns to the concrete tools and technologies that will bring this vision to life. Given the project's reliance on processing and analyzing data, it's imperative that our toolkit not only aligns with these data types but also integrates seamlessly into our existing workflow. Could you provide specific recommendations for tools and technologies tailored to our project's data modeling needs? For each tool recommended, please include: A brief description of how the tool fits into our modeling strategy, particularly in handling our project's data and solution to the pain point. Insights into how each tool integrates with our current technologies. Specific features or modules within these tools that will be most beneficial for our project objectives. Links to official documentation or resources that offer in-depth information about the tool and its use cases relevant to our project. This detailed inquiry aims to ensure that our selection of data modeling tools is not only strategic but also pragmatically focused on enhancing our project's efficiency, accuracy, and scalability.",
        f"As we prepare to test our project's model, we need to generate a large fictitious dataset that mimics real-world data relevant to our project that uses our feature extraction, feature engineering, and metadata management strategies. Generate a python script for creating this dataset and include all attributes from the features needed for our project. The script needs to have recommended tools for dataset creation and validation, compatible with our tech stack, a strategy for incorporating real-world variability and meet our project's model training and validation needs, ensuring the dataset accurately simulates real conditions and integrates seamlessly with our model, enhancing its predictive accuracy and reliability.",
        f"To further refine our approach for our project, we are interested in visualizing an example of the mocked dataset. This example should mimic the real-world data we plan to use, tailored to our project's objectives. Could you provide a sample file that includes: A few rows of data representing what is relevant to our project. An indication of how these data points are structured, including feature names and types. Any specific formatting that will be used for model ingestion, especially concerning its representation for the project. This sample will serve as a visual guide to better understand the mocked data's structure and composition, akin to adding an image for clarity in articles.",
        f"To enhance the project's transition into production, we now seek to develop a production-ready code file for the model(s) utilizing the preprocessed dataset. The focus is on ensuring the code adheres to the high standards of quality, readability, and maintainability observed in large tech companies. Could you provide: A code file or snippet that is structured for immediate deployment in a production environment, specifically designed for our model's data. Detailed comments within the code that explain the logic, purpose, and functionality of key sections, following best practices for documentation. Any conventions or standards for code quality and structure that are commonly adopted in large tech environments, ensuring our codebase remains robust and scalable. This request aims at securing a clear, well-documented, and high-quality code example that can serve as a benchmark for developing our project's production-level machine learning model.",
        f"To effectively deploy the model into production, we require a step-by-step deployment plan that is directly relevant to the unique demands and characteristics of our project, rather than a general deployment plan, including references to necessary tools. Please provide: A brief outline of the deployment steps for our machine learning model, from pre-deployment checks to live environment integration. Key tools and platforms recommended for each step, with direct links to their official documentation. This guide should empower our team with a clear roadmap and the confidence to execute the deployment independently.",
        f"For the final phase of preparing the project for production, we need to create a Dockerfile that encapsulates our environment and dependencies, tailored specifically to our project's performance needs. Please provide: A production-ready Dockerfile with configurations optimized for handling the objectives of our project. Specific instructions within the Dockerfile that address our project's unique performance and scalability requirements. This Dockerfile should serve as a robust container setup, ensuring optimal performance for our specific use case.",
        f"To fully understand the impact of the project: {repository_name}, we need to identify the diverse user groups that will interact with the application. Please provide: A list of the types of users who will benefit from using the application. For each user type, a user story that includes: A brief scenario illustrating their specific pain points. How the application addresses these pain points and the benefits it offers. Which particular file or component of the project facilitates this solution. This information will help showcase the project's wide-ranging benefits and how it serves different audiences, enhancing our understanding of its value proposition.",
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
