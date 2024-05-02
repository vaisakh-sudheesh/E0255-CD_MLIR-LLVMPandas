from datetime import datetime
from pandas_queries import utils

Q_NUM = 2  # Define a new query number for this test case

def q():
    var1 = datetime(1995, 3, 15)  # Example variable, adjust as needed

    customer_ds = utils.get_customer_ds
    orders_ds = utils.get_orders_ds

    customer_ds()
    orders_ds()
    def query():
        nonlocal customer_ds, orders_ds

        customer_ds = customer_ds()
        orders_ds = orders_ds()
        
        # Example join and filtering operation
        joined_data = customer_ds.merge(orders_ds, on='customer_id')
        filtered_data = joined_data[joined_data['order_date'] > var1]

        return filtered_data

    utils.run_query(Q_NUM, query)
if __name__ == "__main__":
    q()
