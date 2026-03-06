# Evaluation Prompts for Monty Spike Experiments

CLI invocation pattern:
```bash
DART_MONTY_LIB_PATH=/Users/runyaga/dev/dart_monty/native/target/release/libdart_monty_native.dylib \
  dart run /Users/runyaga/dev/soliplex-flutter-spike-monty/packages/soliplex_cli/bin/soliplex_cli.dart \
  --host http://localhost:8000 --room spike-120b --monty --prompt "<PROMPT>"
```

## Experiment 0: Baseline 6-Step Pipeline

```
Create a dataset of 10 products with columns: product_name, category (Electronics, Clothing, Food), amount (random between 100-2000), and quantity (1-50). Then: 1) Filter to amount > 500, 2) Group by category with sum of amount, 3) Sort descending by amount, 4) Show top 5 results, 5) Create a bar chart of category vs amount.
```

## Experiment 2: Parameter Extraction

### T1
```
Extract the df_filter parameters from this request: 'filter where amount is greater than 500'. Print the column, operator, and value.
```

### T2
```
Extract the df_filter parameters from this request: 'show me everything over five hundred dollars'. Print the column, operator, and value.
```

### T3
```
Extract the df_filter parameters from this request: 'find products containing pro'. Print the column, operator, and value.
```

## Experiment 4: Multi-Source Merge Pipeline

```
Create two datasets: (1) sales with columns product_id, product_name, category, quantity - 8 rows with Electronics/Clothing/Food items. (2) pricing with columns product_id, unit_price, cost_price - same 8 product_ids. Merge them on product_id, compute profit as (unit_price - cost_price) * quantity, filter to Q3 products (product_id 5-8), group by category summing profit, and create a bar chart.
```

## Experiment 6: Ambiguous Intent

### T1
```
Show me something interesting about this data: products with names, prices from 10 to 1000, and categories.
```

### T2
```
Analyze sales trends
```

### T3
```
Make a chart
```

### T4
```
Do some data stuff with employees
```

## Experiment 7: Math-Heavy Computation

```
Create employee data with name, department (Engineering, Marketing, Sales), and salary (50000-120000) for 15 employees. Calculate the standard deviation and variance of salary for each department. Show results.
```

## Experiment 8: String Processing

```
Create a list of 20 sentences about technology, nature, and science. Count word frequency across all sentences. Show the top 10 most frequent words, excluding common stop words (the, a, an, is, in, of, to, and, for, it, that, with, on, as, this).
```

## Experiment 9: Schema Mismatch Joins

```
Create two datasets with DIFFERENT column names for the join key: (1) employees with emp_id, name, department - 5 rows. (2) salaries with employee_id, annual_salary, bonus - same 5 people but using employee_id instead of emp_id. Rename the key in one dataset so they match, merge them, then group by department showing average total compensation (salary + bonus).
```

## Experiment 10: Conditional Logic & Branching

### T1
```
Generate the first 20 Fibonacci numbers. Classify each as 'small' (< 100), 'medium' (100-999), or 'large' (>= 1000). Print the number and its classification.
```

### T2/T3
```
Create a grade calculator: given a list of 10 students with names and scores (0-100), assign letter grades (A: 90+, B: 80-89, C: 70-79, D: 60-69, F: below 60). Count how many students got each grade. Print the results.
```

## Experiment 11: Long Pipeline 10+ Steps

```
Build a complete data pipeline: 1) Create 15 products with product_name, category (Tech/Home/Food), base_price, and units_sold. 2) Add a revenue column (base_price * units_sold). 3) Add a margin column (revenue * 0.3 for Tech, 0.2 for Home, 0.15 for Food). 4) Filter to revenue > 1000. 5) Drop the base_price column. 6) Rename margin to gross_profit. 7) Group by category summing revenue and gross_profit. 8) Sort by gross_profit descending. 9) Show top 3 categories. 10) Print a final report with the results.
```
