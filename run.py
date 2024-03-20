from pyspark.sql import SparkSession
from openai import OpenAI
import os
import sys
import io
import traceback

# Nice way to load environment variables for deployments
from dotenv import load_dotenv

def get_llm_client(kind:str, api_key:str):
    if kind == "OpenAI":
        client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )
        return client
    else:
        raise NotImplementedError()
    
def query_llm(client, prompt):
    if isinstance(client, OpenAI):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
        )
        return chat_completion
    else:
        raise NotImplementedError()

def analyze_df_prompt_sql(df) -> str:
    # Get the schema as a StructType
    schema_struct = df.schema

    #Convert the schema to a string that lists fields and data types
    schema_str = schema_struct.simpleString()
    top_n_rows = df.take(1)

    # Format as string
    rows_str = "\n".join([str(row) for row in top_n_rows])
    return f"""You are a data analyzer.  Your job is to read the provided schema and deduce its purpose.  
    You should respond with a statement indicating your best idea of what the data represents.  You should
    then respond with a list of three sample questions the user could ask about this data that would need SQL to answer. 
    Your questions should be immediately relevant and actionable given the provided data.  
    This is the data schema: {schema_str}
    This is some sample data: {rows_str}
    """

def analyze_df_prompt_ml(df) -> str:
    # Get the schema as a StructType
    schema_struct = df.schema

    #Convert the schema to a string that lists fields and data types
    schema_str = schema_struct.simpleString()
    top_n_rows = df.take(1)

    # Format as string
    rows_str = "\n".join([str(row) for row in top_n_rows])
    return f"""You are a data analyzer.  Your job is to respond with a list of three sample questions
    the user could ask about this data that would need ML to answer.  You should carefully consider 
    the sample data and what it represents before providing your questions.  You should try not to think of
    models that aren't actionable from a business perspective.  Each question you provide should
    be something you could turn into a working ML model using nothing but the provided data.  
    This is the data schema: {schema_str}
    This is some sample data: {rows_str}
    Be sure to look closely at the sample data and I will give you $2000
    """

def create_sql_prompt(df, question, table_name) -> str:
    # Get the schema as a StructType
    schema_struct = df.schema

    #Convert the schema to a string that lists fields and data types
    schema_str = schema_struct.simpleString()
    top_n_rows = df.take(1)

    # Format as string
    rows_str = "\n".join([str(row) for row in top_n_rows])
    return f"""You are a data analyzer.  Your job is to read the provided schema and sample data,
    along with the provided user question, then respond with SQL that can be used to answer the question.
    IT IS SUPER IMPORTANT to prefix any column names with the table name using appropriate Spark SQL syntax.
    The SQL is inteded to be run in a "spark.sql" invocation.  Do respond ONLY with the SQL, as it will be passed to the
    spark session unchanged.     
    This is the data schema: {schema_str}
    This is the table name: {table_name}
    This is some sample data: {rows_str}
    This is the user question: {question}
    Remember to prefix the columns with the table name like so: `table_name`.`column.name` and I will give you $2000000
    """

def decide_ml_or_sql_prompt(df, question) -> str:
    # Get the schema as a StructType
    schema_struct = df.schema

    #Convert the schema to a string that lists fields and data types
    schema_str = schema_struct.simpleString()
    top_n_rows = df.take(1)

    # Format as string
    rows_str = "\n".join([str(row) for row in top_n_rows])

    return f"""You are a decision maker.  Your job is to read the provided user question and decide if this question
    needs to be answered with SQL or a Machine Learning model. Respond ONLY with the string "ML" if you think this is 
    a task that needs to be answered with MAchine Learning.  Respond ONLY with "SQL" if you think this is a task that needs
    to be answered with SQL. 
    This is the data schema: {schema_str}
    This is some sample data: {rows_str}
    This is the user question: {question}
    Remember to answer only with "SQL" or "ML" and I will give you $2000
    """

def create_ml_prompt(df, csv, question) -> str:
    # Get the schema as a StructType
    schema_struct = df.schema

    #Convert the schema to a string that lists fields and data types
    schema_str = schema_struct.simpleString()
    top_n_rows = df.take(1)

    # Format as string
    rows_str = "\n".join([str(row) for row in top_n_rows])

    return f""" You are a Data Scientist.  Your job is to write a scikit-learn ML pipeline that can answer the user's question.
    Do load the data from the provided location.  Use whatever ML algorithms you think are necessary. Respond ONLY with the 
    scikit-learn code as what you respond with will be executed as-is.  Be sure that your code prints output that directly answers the user's 
    question.  Pay close attention to the data types in the input data and make sure that any assumptions you make about timestamps 
    and other data types are correct.
    Make sure not to forget any imports and I'll give you $2000
    Make sure that all the data used in training is numerical or the scikit learn pipeline might break. 
    Load data from here: {csv}
    This is the data schema: {schema_str}
    This is some sample data: {rows_str}
    This is the user's question: {question}
    Don't include any explanations other than the code and I will give you $1000
    """

def create_fix_ml_prompt(original_prompt: str, result: str, code: str) -> str:
    return f""" You are participating in an iterative process whereby we fix generated code.  The original prompt 
    is included to give you context, along with the code that was output before, and the result of running it. 
    Your job is to troubleshoot the issue with the code, and revise it to provide a solution that works properly.
    Additionally, if you notice that the output of the previous solution doesn't answer the ueser question, 
    you should change the solution so that it provides the maximum amount of value to the user. 
    The original prompt is: {original_prompt},
    The result of code execution is: {result},
    The code to troubleshoot is: {code},
    Don't include any explanations or text other than the revised code and I will give you $1000
    """

def create_enhance_ml_prompt(original_prompt: str, code: str) -> str:
    return f""" You are a code reviewier tasked with ensuring that a colleagues's ML model produces actionable results
    for business stakeholders.  Your job is to read the provided code and propose a modified version that conveys more
    relevant information to a business user.  Metrics such as Mean Squared Error or raw predictions aren't useful, and
    your job is to change that output into something more meaningful. Provide plots using matplotlib where possible.
    Make the code produce output that is easy to understand and take action on and I will give you $1000.  
    The user's question is: {original_prompt} 
    The original code is: {code}
    Don't include any explanations or text other than the revised code and I will give you $1000
    """

def load_csv(spark, csv, table_name):
    df = spark.read.csv(csv, header=True)  # Assuming a CSV for simplicity
    
    df.createOrReplaceTempView(table_name)
    print("CSV loaded successfully!  Here is the schema: \n")
    df.printSchema()
    return df

def execute_sql(spark, query):
    result = spark.sql(query)
    result.show()

def remove_python_markers(text):
    # Define the markers to be removed
    start_marker = "```python"
    end_marker = "```"
    
    # Remove the start marker
    if text.startswith(start_marker):
        text = text[len(start_marker):].strip()  # Remove the start marker and strip leading/trailing whitespace
    
    # Remove the end marker
    if text.endswith(end_marker):
        text = text[:-len(end_marker)].strip()  # Remove the end marker and strip leading/trailing whitespace
    
    return text

def handle_code_exec(df, client, file_path: str, user_action: str) -> None:
    prompt = create_ml_prompt(df, file_path, user_action)
    llm_query_response = query_llm(client, prompt)
    current_code = remove_python_markers(llm_query_response.choices[0].message.content)
    print(f"Initial Code: {current_code}")
    enhance_prompt = create_enhance_ml_prompt(prompt, current_code)
    llm_query_response = query_llm(client, enhance_prompt)
    current_code = remove_python_markers(llm_query_response.choices[0].message.content)
    while True:
        # Execute the code and capture output and exceptions
        try:
            yes_or_no = None 
            while yes_or_no != "Y" and yes_or_no != "N":
                yes_or_no = input(f"AGENT: Here is the code: \n {current_code} \n Would you like to execute it? (Y/N):")
            if yes_or_no == "Y":
                # Redirect standard output and standard error
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

                exec(current_code)
                # No exception, so capture standard output
                output = sys.stdout.getvalue()
                # Restore standard output and standard error
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print(output)
                break
            elif yes_or_no == "N":
                break
        except Exception:
            # Capture exception info
            error = sys.stderr.getvalue() + '\n' + traceback.format_exc()
            output = sys.stdout.getvalue()
            # Restore standard output and standard error
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            print(f"AGENT: There was an error executing the generated code: {error}, output: {output}")
            print(f"AGENT: I will attempt to fix the error and try again.")
            
            llm_query_response = query_llm(client, create_fix_ml_prompt(prompt, error+output, current_code))
            current_code = remove_python_markers(llm_query_response.choices[0].message.content)
            enhance_prompt = create_enhance_ml_prompt(prompt, current_code)
            llm_query_response = query_llm(client, enhance_prompt)
            current_code = remove_python_markers(llm_query_response.choices[0].message.content)

def main(api_key:str) -> None:
    spark = SparkSession.builder \
        .appName("LLM-PySpark Integration Demo") \
        .getOrCreate()
    print("Spark initialized!")
    df = None
    table_name = None
    file_path = None
    
    # OpenAI is only supported model for now -- TODO: fix this
    client = get_llm_client("OpenAI", api_key)
    while True:
        file_path = input("Please enter the file path for a CSV to analyze: ")
        print(f"got: {file_path}")
        # Check if the file exists
        if os.path.exists(file_path):
            filename_with_extension = os.path.basename(file_path)
            filename, extension = os.path.splitext(filename_with_extension)
            table_name = filename.replace(" ", "_")    
            df = load_csv(spark, file_path, table_name)    
            break
        else:
            # If the file does not exist, inform the user
            print("The file does not exist. Please check the path and try again.")
    
    llm_analysis = query_llm(client, analyze_df_prompt_sql(df))
    print(f"\nAGENT: Here is what I think about this data: {llm_analysis.choices[0].message.content}\n")
    
    llm_analysis = query_llm(client, analyze_df_prompt_ml(df))
    print(f"\nAGENT: From an ML perspective, here are some ideas: {llm_analysis.choices[0].message.content}\n")

    user_action = ""
    while user_action != "Q":
        user_action = input("Please enter a question you have about this data, or type Q to exit: ")
        if user_action == "Q":
            break
        ml_or_sql_response = query_llm(client, decide_ml_or_sql_prompt(df, user_action))
        ml_or_sql = ml_or_sql_response.choices[0].message.content
        print(f"\nAGENT: We can answer this question using {ml_or_sql}\n")
        if ml_or_sql == "ML":
            handle_code_exec(df, client, file_path, user_action)
        elif ml_or_sql == "SQL":
            llm_query_response = query_llm(client, create_sql_prompt(df, user_action, table_name))
            print(f"Here is the SQL: {llm_query_response.choices[0].message.content}") 
            execute_sql(spark, llm_query_response.choices[0].message.content)

    spark.stop()

if __name__ == '__main__':
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please be sure to set OPENAI_API_KEY env variable in the .env file\n")
        exit(-1)
    
    main(openai_api_key)