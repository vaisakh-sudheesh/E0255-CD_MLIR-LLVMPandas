from datetime import datetime
from pandas_queries import utils
import pandas as pd

Q_NUM = 3  # Define a new query number for this test case

def q():

    customer_ds = utils.get_customer_ds
    orders_ds = utils.get_orders_ds
    suppliers_ds = utils.get_supplier_ds
    regions_ds = utils.get_region_ds
    
    customer_ds()
    regions_ds()
    orders_ds()
    suppliers_ds()

    def query():
        nonlocal customer_ds, orders_ds, suppliers_ds, regions_ds
        
        customer_ds = customer_ds()
        regions_ds = regions_ds()
        orders_ds = orders_ds()
        suppliers_ds = suppliers_ds()
       
        # Join orders with customers
        cust_orders = pd.merge(orders_ds, customer_ds, on='c_custkey')
        
        # Join the result with suppliers
        order_suppliers = pd.merge(cust_orders, suppliers_ds, left_on='o_custkey', right_on='s_suppkey')
        
        # Final join with regions
        full_join = pd.merge(order_suppliers, regions_ds, left_on='s_nationkey', right_on='r_regionkey')
        
        # Filter orders by a certain total price and recent dates
        recent_high_value_orders = full_join[
            (full_join['o_totalprice'] > 30000) &
            (full_join['o_orderdate'] > pd.Timestamp('2020-01-01'))
        ]
        
        # Aggregate to get total prices per region
        total_price_per_region = recent_high_value_orders.groupby('r_name').agg({
            'o_totalprice': 'sum'
        }).reset_index()
        
        return total_price_per_region

    utils.run_query(Q_NUM, query)
if __name__ == "__main__":
    q()
