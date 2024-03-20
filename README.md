# sparkalyzer
Use the power of LLMs to write your Spark SQL for you!  Also, generate some ML stuff and see if it works!
*Relies on unguarded dynamic code execution from an LLM -- use at own risk!* 

## Installation
First, install python dependencies

```shell
pip3 install -r requirements.txt
```

Next, rename `sample.env` to `.env` and change the line that says `OPENAI_API_KEY=<your API key>` to whatever your key is.

In the future I will make it so you can use other models, pinky swear. 

## Usage
In a terminal, run:
```shell
python3 run.py
```
When the prompt asks, paste in the location of an interesting csv file that you want to analyze.  Then type your question and watch the show. 

## Known Issues
Since this is using an LLM to generate SQL code / ML code, there is absolutely no guarantee that what it spits out will
*actually* work.  Use at your own risk.

That being said, I have seen a few issues where column names have spaces and the LLM doesn't pick up on those.  

