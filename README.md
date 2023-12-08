# Market Basket Analysis

Hello everyone, in this project we will analyze product associations in customer shopping carts using synthetic data to develop sales strategies for retail businesses and recommend some actions. The analysis is performed using Apriori algorithm to understand customer behavior, improve recommendation systems and provide valuable information for various strategies such as promotions, campaigns, etc. The project was implemented using Python programming language and related libraries."

![1_DHfQvlMVBaJCHpYmj1kmCw](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/698a1da4-ff40-4d6d-9bc7-0ed4b4e6f925)


## WHAT IS MARKET BASKET ANALYSIS?

Market basket analysis is a strategic data mining technique used by retailers to enhance sales by gaining a deeper understanding of customer purchasing patterns. This method entails the examination of substantial datasets, such as historical purchase records, in order to unveil inherent product groupings and identify items that tend to be bought together.[1]

Let's start coding...

## STEP 1: Create a Dataset to use it.

There are many public datasets to use in this project. But in this project I will create a synthetic datafarame using the "random" module from Python.

First of all, in our data set, we will have these columns:

----------------------------------------------------------------------------------

Customer_ID: A unique identifier for each customer.

OrderNumber: A unique identifier for each order.

OrderDate: The date when the order was placed

OrderTime: The time when the order was placed

District: The market branch in Istanbul where the customer makes a purchase.

PaymentMethod: The method used for payment (Credit Card, Debit Card, Cash).

Items: A list of items in each order, separated by commas.

----------------------------------------------------------------------------------

I also created a total of 10000 rows of data. I set some rules in this data set.

- There are about 4000 unique customers.
- There are 17 different product types.
- There can be at least 3 and at most 10 different products in a basket.
- A customer can shop at most 4 times.
- There are 3 different payment types
- These purchases must be made in one of the stores located in 10 different neighborhoods in Istanbul.

The entire data set is randomly generated using these variables.

Yes now we have enough information about the dataset, we can analyze the data.

First of all we should import the necessary libraries.

![Screenshot 2023-12-08 143349](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/8318da3c-ac00-402e-8a54-1c796c0e7d98)

Then we prepare lists of payment methods, districts and products and create a empty datafarame with column names.

![Screenshot 2023-12-08 143457](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/54dc07df-10a8-4948-89b3-aac08efcbe30)

After that, we set our rules for total number of rows, , maximum number of items per basket, maximum number of transaction per customer to create dataset.

![Screenshot 2023-12-08 143716](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/a8afc7b0-01cd-418c-89fb-e86d798c3e74)

We add random order details and create the dataset.

![Screenshot 2023-12-08 143943](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/a3c0d8d6-d3a6-4ed5-b137-83972bba36a3)

And finally we edit the column names and save our data as a csv file.

![Screenshot 2023-12-08 144324](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/789190cc-c0fd-49ee-8801-0e0a91ea6b3d)

Let's look at the first 5 rows of our data.

![Screenshot 2023-12-08 144337](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/11479292-ed89-4abe-87f8-bbf520eb87fa)

Let's take a look at the variables

![Screenshot 2023-12-08 144402](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/836a8b85-c5a8-45d4-91af-cd28ec842a9a)

## STEP 2: Prepare the data for analysis.

At this point, we will prepare the data for analysis. First of all, as you can see in the Items column, there are commas between each product. We should seperate each product to use in the A-priori Method. 

![Screenshot 2023-12-08 145409](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/b7022138-1aa4-4b24-82d6-bf076afc445c)

As you can see above, it returns us a list of items in each customer's shopping basket.

Now we need to transform our data into a table of boolean values as required by the A-priori algorithm. We have a pratical way to do this, we use the TransactionEncoder() function from mlxtend module to transform the data.

And after that let's see the transformed data. 

![Screenshot 2023-12-08 165436](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/2a73d0eb-5a13-4004-9cda-2386a4d902b8)

As you can see above, we have converted every single transaction row into boolean form to use in the a-priori algorithm. False means that there is no product specified in the transaction in that row. And Yes means that there is.

## STEP 3: A-Priori Algorithm and Interpret Results

At this point, we should know some kind of terms about the algorithm and analysis.

Support:

Explanation: It measures how frequently a product or product group appears in total shopping baskets.

Example: If the "BREAD" product is found in 20% of total shopping baskets, the support value for "BREAD" is 20%.

Confidence:
Explanation: It indicates the relationship between two products, specifying the probability of the second product being purchased when the first product is purchased.

Example: If the probability of purchasing "COOKIES" when "MILK" is purchased is 30%, the confidence value is 30%.

Lift:
Explanation: It shows how much more likely the relationship between two products occurs compared to a random situation.

Example: If the probability of purchasing "BISCUITS" when "TEA" is purchased is 2, the lift value is 2, indicating that this relationship occurs twice as much as in a random situation.
These terms are crucia

Yes, we can now use the a-priori algorithm to see how often each product is involved in transactions. We set the support value to 15%.

![Screenshot 2023-12-08 170013](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/b4e7f3ec-e348-42e5-a382-140ad4093dda)

Let's interpret this results.

BREAD 0.3853 (38%) has a support value of 0.3853, meaning that this product alone was involved in 38% of all purchases. There are 10000 rows of transactions in our data set, each representing a different purchase. Also, there can only be one of each product in each transaction. This means that we sold a total of 3853 loaves of bread.

BREAD is followed by BUTTER, COFFEE and COKE.

We can see the support values for each product with this method. Now, using the association_rules() function, we will extract the associations of products purchased together from the product associations obtained with the Apriori algorithm. Let's run the code and see the results.

![Screenshot 2023-12-08 173102](https://github.com/enesbesinci/CRM-market-basket-analysis/assets/110482608/b2a9cf94-72c8-465b-b930-032d6bb7edee)

This table tells a lot of useful things for our retail business.

Let's discover them.

### Product Placement and Shelf Layout:

The analysis shows that there is a strong association between "BUTTER" and "BREAD", so displaying these products together or placing them side by side can encourage customers to shop.

### Promotions and Discounts:

For example, there is a strong association between "JAM" and "BREAD", we can offer these products together with a special discount or promotion. We can organize various promotions to encourage customers to buy a certain product.

### Inventory Management:

Market Basket Analysis helps with demand forecasting. If a certain product is often sold together with another product, we can optimize inventory management by taking this into account.

### Data Collection and Analysis:

By continuously monitoring and analyzing customer shopping data, we can identify new relationships and opportunities. Understanding customer behavior helps you continuously update our strategy.




