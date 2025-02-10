You are building a Bank statement analyzer. You have to build a Node.js (or Python) script which 
a. Accepts one of the provided .txt files as input
b. Extracts individual transactions from the provided txt file
c. Converts each transaction into a json object of the form:
{ 
 "date": Date of transaction in DD/MM/YYYY format
 "narration": Complete Description of transaction
 "amount": Amount of transaction
 "type": Debit or Credit 
}
d. Prints the JSON.

In the folder, you will find some sample txt files to test against.
Ideally, your code should be able to handle any such file.
