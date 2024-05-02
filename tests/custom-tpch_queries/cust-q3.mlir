module {

    llvm.func @main() {
    //##     customer_ds = utils.get_customer_ds
    //##     orders_ds = utils.get_orders_ds
    //##     suppliers_ds = utils.get_supplier_ds
    //##     regions_ds = utils.get_region_ds
    //##     
    //##     customer_ds()
    //##     regions_ds()
    //##     orders_ds()
    //##     suppliers_ds()
    //##     
    //##     def query():
    //##         nonlocal customer_ds, orders_ds, suppliers_ds, regions_ds
    //##     
    //##         customer_ds = customer_ds()
    //##         regions_ds = regions_ds()
    //##         orders_ds = orders_ds()
    //##         suppliers_ds = suppliers_ds()
            
            %customer_ds = pandas.read_csv("customer_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/customer.csv") : !llvm.ptr
            %region_ds   = pandas.read_csv("region_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/region.csv") : !llvm.ptr
            %orders_ds   = pandas.read_csv("orders_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/orders.csv") : !llvm.ptr
            %supplier_ds = pandas.read_csv("supplier_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/supplier.csv") : !llvm.ptr

    //>>  Description:
    //>>  Merging the orders_ds DataFrame with the customer_ds DataFrame using the pd.merge() function. 
    //>>  The merge is performed based on the common column 'c_custkey'. The result is stored in the cust_orders DataFrame.
    //>>  
    //>>    WARN: THERE IS NO COMMON COLUMN BETWEEN orders_ds, customer_ds with name c_custkey, hence skipping this.
    //>>
    //##         # Join orders with customers
    //##         cust_orders = pd.merge(orders_ds, customer_ds, on='c_custkey')
            //%cust_orders = pandas.merge_on(%orders_ds : !llvm.ptr, %customer_ds : !llvm.ptr, "c_custkey")  !llvm.ptr


    //>>  Description
    //>>  Merges the cust_orders DataFrame with the suppliers_ds DataFrame using the pd.merge() function again.
    //>>  This time, the merge is performed based on the columns 'o_custkey' from cust_orders and 's_suppkey' from suppliers_ds.
    //>>  The result is stored in the order_suppliers DataFrame.      
    //>> 
    //##         # Join the result with suppliers
    //##         order_suppliers = pd.merge(cust_orders, suppliers_ds, left_on='o_custkey', right_on='s_suppkey')
            %order_suppliers = pandas.merge_lr(%cust_orders, %suppliers_ds, "o_custkey","s_suppkey")  !llvm.ptr


    //>>  Description:
    //>>  Then, the code performs another merge operation by joining the order_suppliers DataFrame with the regions_ds DataFrame.
    //>>  The merge is based on the columns 's_nationkey' from order_suppliers and 'r_regionkey' from regions_ds.
    //>>  The result is stored in the full_join DataFrame.
    //>>
    //##         # Final join with regions
    //##         full_join = pd.merge(order_suppliers, regions_ds, left_on='s_nationkey', right_on='r_regionkey')
            %full_join = pandas.merge_full(%order_suppliers, %regions_ds, "s_nationkey","r_regionkey")  !llvm.ptr


    //>>  Description:
    //>>  After the merges, the code filters the full_join DataFrame to select only the rows that meet certain conditions.
    //>>  Specifically, it selects rows where the 'o_totalprice' column is greater than 30000 and the 'o_orderdate' column is 
    //>>  greater than January 1, 2020 (represented as a pd.Timestamp object). 
    //>>  The filtered DataFrame is stored in the recent_high_value_orders variable.
    //>>
    //##         # Filter orders by a certain total price and recent dates
    //##         recent_high_value_orders = full_join[
    //##             (full_join['o_totalprice'] > 30000) &
    //##             (full_join['o_orderdate'] > pd.Timestamp('2020-01-01'))
    //##         ]
            %full_join1               = pandas.filter_num_gt(%full_join,  "o_totalprice", 30000)         !llvm.ptr
            %recent_high_value_orders = pandas.filter_date_gt(%full_join1,"o_orderdate" , "2020-01-01")  !llvm.ptr


    //>>  Description:
    //>>  The code then performs an aggregation operation on the recent_high_value_orders DataFrame using the groupby() function.
    //>>  It groups the data by the 'r_name' column and calculates the sum of the 'o_totalprice' column for each group.
    //>>  The result is stored in the total_price_per_region DataFrame.
    //##         # Aggregate to get total prices per region
    //##         total_price_per_region = recent_high_value_orders.groupby('r_name').agg({
    //##             'o_totalprice': 'sum'
    //##         }).reset_index()
            %total_price_per_region   = pandas.groupby_sum(%recent_high_value_orders, "r_name", "o_totalprice")  !llvm.ptr



    //##         
    //##         return total_price_per_region
            %nrows = llvm.mlir.constant(5 : index) : i32
            %dummy = pandas.print_df (%total_price_per_region, %nrows)  : (!llvm.ptr, i32) !llvm.ptr

        llvm.return
    }
}