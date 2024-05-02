module {

    llvm.func @main() {
    //##    date1 = datetime.datetime.strptime("1994-01-01", "%Y-%m-%d")
    //##    date2 = datetime.datetime.strptime("1995-01-01", "%Y-%m-%d")
            %date1 = pandas.format_ts ("1994-01-01")
            %date2 = pandas.format_ts ("1995-01-01")
    //##
    //##    region_ds = utils.get_region_ds
    //##    nation_ds = utils.get_nation_ds
    //##    customer_ds = utils.get_customer_ds
    //##    line_item_ds = utils.get_line_item_ds
    //##    orders_ds = utils.get_orders_ds
    //##    supplier_ds = utils.get_supplier_ds
    //##
    //##    # first call one time to cache in case we don't include the IO times
    //##    region_ds()
    //##    nation_ds()
    //##    customer_ds()
    //##    line_item_ds()
    //##    orders_ds()
    //##    supplier_ds()
    //##
    //##    def query():
    //##        nonlocal region_ds
    //##        nonlocal nation_ds
    //##        nonlocal customer_ds
    //##        nonlocal line_item_ds
    //##        nonlocal orders_ds
    //##        nonlocal supplier_ds
    //##
    //##        region_ds = region_ds()
    //##        nation_ds = nation_ds()
    //##        customer_ds = customer_ds()
    //##        line_item_ds = line_item_ds()
    //##        orders_ds = orders_ds()
    //##        supplier_ds = supplier_ds()
            %region_ds = pandas.read_csv("region_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/region.csv") : !llvm.ptr
            %nation_ds   = pandas.read_csv("nation_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/nation.csv") : !llvm.ptr
            %customer_ds = pandas.read_csv("customer_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/customer.csv") : !llvm.ptr
            %line_item_ds = pandas.read_csv("line_item_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/lineitem.csv") : !llvm.ptr            
            %orders_ds   = pandas.read_csv("orders_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/orders.csv") : !llvm.ptr
            %supplier_ds   = pandas.read_csv("supplier_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/supplier.csv") : !llvm.ptr

    //##  Description: 
    //##  Selecting rows from the region_ds dataframe where the value in the column r_name is equal to "ASIA". 
    //##  This creates a boolean mask rsel for filtering the rows.
    //##        rsel = region_ds.r_name == "ASIA"
            %region_str = "ASIA"
            %rsel       = pandas.filter_eq(%region_ds : !llvm.ptr, "r_name", %region_str)

    //##  Description:
    //##  Selects rows from the orders_ds dataframe where the value in the column o_orderdate is greater than or equal to date1 and 
    //##  less than date2. This creates a boolean mask osel for filtering the rows
    //##        osel = (orders_ds.o_orderdate >= date1) & (orders_ds.o_orderdate < date2)
    //##        forders = orders_ds[osel]
            osel_1 = pandas.filter_lt(%orders_ds : !llvm.ptr, "r_name", %date2)
            osel   = pandas.filter_geq(%osel_1 : !llvm.ptr, "r_name", %date1)

    //##  Description:
    //##  Applies the boolean masks osel and rsel to filter the rows in the orders_ds and region_ds dataframes, respectively. 
    //##  The filtered dataframes are assigned to forders and fregion.
    //##        fregion = region_ds[rsel]

    //##  Description:
    //##  Performs a series of merges using the merge() function to combine the filtered dataframes with other dataframes 
    //##  (nation_ds, customer_ds, line_item_ds, and supplier_ds) based on specific column matches.
    //##  Each merge operation creates a new dataframe (jn1, jn2, jn3, jn4, jn5) that contains the combined data.
    //##        jn1 = fregion.merge(nation_ds, left_on="r_regionkey", right_on="n_regionkey")


    //##        jn2 = jn1.merge(customer_ds, left_on="n_nationkey", right_on="c_nationkey")


    //##        jn3 = jn2.merge(forders, left_on="c_custkey", right_on="o_custkey")


    //##        jn4 = jn3.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")


    //##        jn5 = supplier_ds.merge(
    //##            jn4,
    //##            left_on=["s_suppkey", "s_nationkey"],
    //##            right_on=["l_suppkey", "n_nationkey"],
    //##        )

    //##  Description:
    //##  After the merges, a new column revenue is created in the jn5 dataframe by multiplying the values in the 
    //##  l_extendedprice and l_discount columns.
    //##        jn5["revenue"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)

    //##  Description:
    //##  Finally, the gb dataframe is sorted in descending order based on the values in the revenue column, and 
    //##  the sorted dataframe is assigned to result_df.
    //##        gb = jn5.groupby("n_name", as_index=False)["revenue"].sum()

    
    //##        result_df = gb.sort_values("revenue", ascending=False)

    //##        return result_df


        llvm.return
    }
}