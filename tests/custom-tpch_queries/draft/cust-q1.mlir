module {

    llvm.func @main() {
    
    //>>  Description:
    //>>  Load the dataframes from the respective CSV files.
    //>>
    //##     nation_ds = utils.get_nation_ds
    //##     customer_ds = utils.get_customer_ds
    //##     line_item_ds = utils.get_line_item_ds
    //##     orders_ds = utils.get_orders_ds
    //##     supplier_ds = utils.get_supplier_ds
    //##     # first call one time to cache in case we don't include the IO times
    //##     nation_ds()
    //##     customer_ds()
    //##     line_item_ds()
    //##     orders_ds()
    //##     supplier_ds()
    //##
    //##     def query():
    //##          nonlocal nation_ds
    //##          nonlocal customer_ds
    //##          nonlocal line_item_ds
    //##          nonlocal orders_ds
    //##          nonlocal supplier_ds
    %nation_ds   = pandas.read_csv("nation_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/nation.csv") : !llvm.ptr
    %customer_ds = pandas.read_csv("customer_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/customer.csv") : !llvm.ptr
    %line_item_ds = pandas.read_csv("line_item_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/lineitem.csv") : !llvm.ptr
    %orders_ds   = pandas.read_csv("orders_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/orders.csv") : !llvm.ptr
    %supplier_ds   = pandas.read_csv("supplier_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/supplier.csv") : !llvm.ptr

    //>>  Description: 
    //>>  Filtering the line_item_ds dataframe based on the "l_shipdate" column. It selects rows where the ship date is between 
    //>>  January 1, 1995, and January 1, 1997.
    //>>
    //##          lineitem_filtered = line_item_ds[
    //##              (line_item_ds["l_shipdate"] >= datetime(1995, 1, 1))
    //##              & (line_item_ds["l_shipdate"] < datetime(1997, 1, 1))
    //##          ]
    %lineitem_filtered_1 = pandas.filter_date_geq(%line_item_ds : !llvm.ptr, "l_shipdate", "1995-1-1 T00:00:00.000000" , "lineitem_filtered_1") !llvm.ptr
    %lineitem_filtered = pandas.filter_date_lt(%lineitem_filtered_1 : !llvm.ptr, "l_shipdate", "1997-1-1 T00:00:00.000000" , "lineitem_filtered_1") !llvm.ptr

    //>>  Description:
    //>>  Two new columns are added to the lineitem_filtered dataframe. 
    //>>  The "l_year" column is created by extracting the year from the "l_shipdate" column using the dt.year accessor.
    //>>
    //##          lineitem_filtered["l_year"] = lineitem_filtered["l_shipdate"].dt.year
    

    //>>  Description:
    //>>   The "revenue" column is calculated by multiplying the "l_extendedprice" column with (1 - "l_discount").
    //>>
    //##          lineitem_filtered["revenue"] = lineitem_filtered["l_extendedprice"] * (
    //##              1.0 - lineitem_filtered["l_discount"]
    //##          )

    //>>  Description:
    //>>   e lineitem_filtered dataframe is then further filtered to include only specific columns: "l_orderkey", "l_suppkey", "l_year", and "revenue".
    //>>
    //##          lineitem_filtered = lineitem_filtered.loc[
    //##              :, ["l_orderkey", "l_suppkey", "l_year", "revenue"]
    //##          ]

    //>>  Description:
    //>>  The supplier_ds, orders_ds, and customer_ds dataframes are filtered to include specific columns.
    //>>
    //##          supplier_filtered = supplier_ds.loc[:, ["s_suppkey", "s_nationkey"]]
    //##          orders_filtered = orders_ds.loc[:, ["o_orderkey", "o_custkey"]]
    //##          customer_filtered = customer_ds.loc[:, ["c_custkey", "c_nationkey"]]

    //>>  Description:
    //>>  Two new dataframes, n1 and n2, are created by filtering the nation_ds dataframe based on the "n_name" column. 
    //>>  One dataframe contains rows where the nation name is "FRANCE", and the other contains rows where the nation name is "GERMANY"
    //>>
    //##          n1 = nation_ds[(nation_ds["n_name"] == "FRANCE")].loc[
    //##              :, ["n_nationkey", "n_name"]
    //##          ]
    //##          n2 = nation_ds[(nation_ds["n_name"] == "GERMANY")].loc[
    //##              :, ["n_nationkey", "n_name"]
    //##          ]


    //>>  Description:
    //>>  Performs a series of merges and transformations on the filtered dataframes. 
    //>>  It starts with the customer_filtered dataframe and merges it with the n1 dataframe based on the "c_nationkey" and "n_nationkey" columns.
    //>>  The resulting dataframe, N1_C, drops unnecessary columns and renames the "n_name" column to "cust_nation".
    //>>
    //##          # ----- do nation 1 -----
    //##          N1_C = customer_filtered.merge(
    //##              n1, left_on="c_nationkey", right_on="n_nationkey", how="inner"
    //##          )


    //##          N1_C = N1_C.drop(columns=["c_nationkey", "n_nationkey"]).rename(
    //##              columns={"n_name": "cust_nation"}
    //##          )

    //>>  Description:
    //>>  Next, N1_C is merged with the orders_filtered dataframe based on the "c_custkey" and "o_custkey" columns.
    //>>  The resulting dataframe, N1_C_O, drops unnecessary columns.
    //>>
    //##          N1_C_O = N1_C.merge(
    //##              orders_filtered, left_on="c_custkey", right_on="o_custkey", how="inner"
    //##          )
    //##          N1_C_O = N1_C_O.drop(columns=["c_custkey", "o_custkey"])

    //>>  Description:
    //>>  N2_S is then merged with the lineitem_filtered dataframe based on the "s_suppkey" and "l_suppkey" columns. 
    //>>  The resulting dataframe, N2_S_L, drops unnecessary columns.
    //>>
    //##          N2_S = supplier_filtered.merge(
    //##              n2, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    //##          )
    //##          N2_S = N2_S.drop(columns=["s_nationkey", "n_nationkey"]).rename(
    //##              columns={"n_name": "supp_nation"}
    //##          )

    //>>  Description:
    //>>  N2_S is then merged with the lineitem_filtered dataframe based on the "s_suppkey" and "l_suppkey" columns.
    //>>  The resulting dataframe, N2_S_L, drops unnecessary columns.
    //>>
    //##          N2_S_L = N2_S.merge(
    //##              lineitem_filtered, left_on="s_suppkey", right_on="l_suppkey", how="inner"
    //##          )
    //##          N2_S_L = N2_S_L.drop(columns=["s_suppkey", "l_suppkey"])


    //>>  Description:
    //>>  The total1 dataframe is created by merging N1_C_O with N2_S_L based on the "o_orderkey" and "l_orderkey" columns. 
    //>>  Unnecessary columns are dropped.
    //>>
    //##          total1 = N1_C_O.merge(
    //##              N2_S_L, left_on="o_orderkey", right_on="l_orderkey", how="inner"
    //##          )
    //##          total1 = total1.drop(columns=["o_orderkey", "l_orderkey"])


    //>>  Description:
    //>>  The same set of operations is repeated for the N2_C_O dataframe, which is created by merging customer_filtered with n2, and
    //>>  then merging with N1_S_L.
    //>>
    //##          # ----- do nation 2 ----- (same as nation 1 section but with nation 2)
    //##          N2_C = customer_filtered.merge(
    //##              n2, left_on="c_nationkey", right_on="n_nationkey", how="inner"
    //##          )
    //##          N2_C = N2_C.drop(columns=["c_nationkey", "n_nationkey"]).rename(
    //##              columns={"n_name": "cust_nation"}
    //##          )
    //##          N2_C_O = N2_C.merge(
    //##              orders_filtered, left_on="c_custkey", right_on="o_custkey", how="inner"
    //##          )
    //##          N2_C_O = N2_C_O.drop(columns=["c_custkey", "o_custkey"])
    //##
    //##          N1_S = supplier_filtered.merge(
    //##              n1, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    //##          )
    //##          N1_S = N1_S.drop(columns=["s_nationkey", "n_nationkey"]).rename(
    //##              columns={"n_name": "supp_nation"}
    //##          )
    //##          N1_S_L = N1_S.merge(
    //##              lineitem_filtered, left_on="s_suppkey", right_on="l_suppkey", how="inner"
    //##          )
    //##          N1_S_L = N1_S_L.drop(columns=["s_suppkey", "l_suppkey"])

    //>>  Description:
    //>>  The total2 dataframe is created by merging N2_C_O with N1_S_L based on the "o_orderkey" and "l_orderkey" columns.
    //>>  Unnecessary columns are dropped.
    //>>
    //##          total2 = N2_C_O.merge(
    //##              N1_S_L, left_on="o_orderkey", right_on="l_orderkey", how="inner"
    //##          )
    //##          total2 = total2.drop(columns=["o_orderkey", "l_orderkey"])


    //>>  Description:
    //>>  The total1 and total2 dataframes are concatenated using pd.concat() to create the total dataframe.
    //>>
    //##          # concat results
    //##          total = pd.concat([total1, total2])

    //>>  Description:
    //>>  The result_df dataframe is created by grouping the total dataframe by the columns "supp_nation", "cust_nation", and "l_year", and 
    //>>  aggregating the "revenue" column using the sum() function.
    //>>  The resulting dataframe is sorted based on the columns "supp_nation", "cust_nation", and "l_year" in ascending order.
    //>>
    //##          result_df = (
    //##              total.groupby(["supp_nation", "cust_nation", "l_year"])
    //##              .revenue.agg("sum")
    //##              .reset_index()
    //##          )
    //##          result_df.columns = ["supp_nation", "cust_nation", "l_year", "revenue"]
    //##          result_df = result_df.sort_values(
    //##              by=["supp_nation", "cust_nation", "l_year"],
    //##              ascending=[
    //##                  True,
    //##                  True,
    //##                  True,
    //##              ],
    //##          )

    
    //##          return result_df

        llvm.return
    }
}