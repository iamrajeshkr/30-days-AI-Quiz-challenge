Below is a connected, multi-part question set that covers a range of skills in both **pandas** and **numpy**—from reading and merging data, through data transformation and filtering, to performing vectorized computations and addressing performance considerations. You can imagine that the following scenario is presented during an interview:

---

### **Scenario**

You have two CSV files:

1. **`sales_data.csv`**  
   Contains transactional sales data with the columns:
   - `transaction_id` (int)
   - `product_id` (int)
   - `quantity` (int)
   - `price` (float)
   - `date` (string in the format `'YYYY-MM-DD'`)

2. **`products.csv`**  
   Contains product details with the columns:
   - `product_id` (int)
   - `category` (string)
   - `product_name` (string)

---

### **Questions**

1. **Loading Data**  
   *a. Write a Python code snippet to load both CSV files into separate pandas DataFrames.*

2. **Merging DataFrames**  
   *a. Merge the two DataFrames on the `product_id` column using an inner join.  
   b. How would you verify the number of rows in the merged DataFrame?*

3. **Feature Engineering**  
   *a. Create a new column named `total_sales` in the merged DataFrame, calculated as the product of `quantity` and `price`.  
   b. Convert the `date` column to a datetime object using pandas.  
   c. Filter the DataFrame to include only transactions that occurred in the year 2023.*

4. **Aggregation and Grouping**  
   *a. Group the filtered DataFrame by the `category` column. For each category, calculate:  
      - The total quantity sold.  
      - The sum of `total_sales`.  
   b. Identify the category with the highest total sales revenue.*

5. **Vectorized Operations with NumPy**  
   *a. Using numpy, compute the natural logarithm (base e) of the `price` column and add this as a new column called `log_price` to the merged DataFrame.  
   b. Calculate the z-score (i.e., the standard score) for the `total_sales` column using numpy and store the result in a new column called `sales_zscore`.*

6. **Handling Missing Data**  
   *a. Suppose there are missing values in the `quantity` column. Write code to fill these missing values with the median value of the column.*

7. **Performance Considerations and Best Practices**  
   *a. Discuss some potential pitfalls when merging datasets (for example, issues related to duplicate keys or missing values in key columns).  
   b. How would you optimize these pandas/numpy operations if you were dealing with very large datasets (e.g., with millions of rows)?*

---

### **Expectations**

- **Code:** For each coding task, you should be comfortable writing concise, efficient Python code using pandas and numpy.
- **Concepts:** Be prepared to explain why certain functions (like vectorized numpy operations) are preferred over iterative methods.
- **Optimization:** Discuss practical strategies such as using appropriate data types, indexing, and leveraging chunking or Dask when scaling to large datasets.

This series of questions is designed to gauge your practical skills in data manipulation, transformation, and performance tuning using numpy and pandas—key tools for any data scientist.

Good luck, and feel free to ask if you need clarifications on any part!