module {

    llvm.func @main() {
    // Load the CSV data in to dataframes
    //##     customer_ds = utils.get_customer_ds
    //##     orders_ds = utils.get_orders_ds
    //##
    //##     customer_ds()
    //##     orders_ds()
    //##     def query():
    //##        nonlocal customer_ds, orders_ds
    //##
    //##        customer_ds = customer_ds()
    //##        orders_ds = orders_ds()
    %customer_ds = pandas.read_csv("customer_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/customer.csv") : !llvm.ptr
    %orders_ds = pandas.read_csv("orders_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/orders.csv") : !llvm.ptr

    //>>  Description:
    //>>  The code starts by performing a join operation between two Pandas dataframes, customer_ds and orders_ds, using the merge() function.
    //>>  The on='customer_id' parameter specifies that the join should be performed based on the 'customer_id' column in both dataframes.
    //>>  The result of the join operation is stored in a new dataframe called joined_data.
    //>> 
    //##        # Example join and filtering operation
    //##        joined_data = customer_ds.merge(orders_ds, on='customer_id')
    %joined_data_ds = pandas.merge_on (%customer_ds : !llvm.ptr , %orders_ds : !llvm.ptr, "customer_id") !llvm.ptr

    //>>  Description:
    //>>  Applies a filtering operation on the joined_data dataframe. It selects only the rows where the 'order_date' column is greater 
    //>>  than the value of the variable var1. 
    //>>  This filtering is done using boolean indexing, where joined_data['order_date'] > var1 creates a boolean mask indicating which 
    //>>  rows satisfy the condition.
    //>>  The filtered data is then stored in a new dataframe called filtered_data.
    //>>
    //##        var1 = datetime(1995, 3, 15)  # Example variable, adjust as needed
    //##        filtered_data = joined_data[joined_data['order_date'] > var1]
    %filtered_data_ds = pandas.filter_date_gt(%joined_data_ds : !llvm.ptr, "o_orderdate", "1995-3-15T00:00:00.000000" , "joined_data") !llvm.ptr

    //##        return filtered_data
    %nrows = llvm.mlir.constant(10 : index) : i32
    %dummy = pandas.print_df (%filtered_data_ds, %nrows)  : (!llvm.ptr, i32) !llvm.ptr

        llvm.return
    }
}