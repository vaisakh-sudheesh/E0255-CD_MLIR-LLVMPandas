module {

    llvm.func @main() {
    //##    date1 = datetime(1994, 1, 1)
    //##    date2 = datetime(1995, 1, 1)
    //##    var3 = 24
            %date1 = pandas.format_ts ("1994-01-01")
            %date2 = pandas.format_ts ("1995-01-01")
            %var3  = pandas.constant_int(24)
            %var4  = pandas.constant_num(0.05)
            %var5  = pandas.constant_num(0.07)
    //##
    //##    line_item_ds = utils.get_line_item_ds
    //##
    //##    # first call one time to cache in case we don't include the IO times
    //##    line_item_ds()
    //##
    //##    def query():
    //##        nonlocal line_item_ds
    //##        line_item_ds = line_item_ds()
            %line_item_ds = pandas.read_csv("line_item_ds", "/home/vaisakhps/developer/CDProject-v2/data/tables_scale_1/lineitem.csv") : !llvm.ptr

    //>>  Description:
    //>>  Selecting specific columns from the line_item_ds dataframe using the loc function.
    //>>  The selected columns are "l_quantity", "l_extendedprice", "l_discount", and "l_shipdate".
    //>>  The resulting dataframe is stored in the variable lineitem_filtered
    //>>
    //##        lineitem_filtered = line_item_ds.loc[
    //##            :, ["l_quantity", "l_extendedprice", "l_discount", "l_shipdate"]
    //##        ]
            %lineitem_filtered = pandas.subset (%line_item_ds, "l_quantity,l_extendedprice,l_discount,l_shipdate","lineitem_filtered")

    //>>  Description:
    //>>  Next, a boolean condition is created using multiple conditions using the & operator.
    //>>  The conditions are as follows:
    //>>  
    //>>      - lineitem_filtered.l_shipdate >= date1: This condition checks if the "l_shipdate" column is greater than or equal to date1.
    //>>      - lineitem_filtered.l_shipdate < date2: This condition checks if the "l_shipdate" column is less than date2.
    //>>      - lineitem_filtered.l_discount >= 0.05: This condition checks if the "l_discount" column is greater than or equal to 0.05.
    //>>      - lineitem_filtered.l_discount <= 0.07: This condition checks if the "l_discount" column is less than or equal to 0.07.
    //>>      - lineitem_filtered.l_quantity < var3: This condition checks if the "l_quantity" column is less than var3.
    //##        sel = (
    //##            (lineitem_filtered.l_shipdate >= date1)
    //##            & (lineitem_filtered.l_shipdate < date2)
    //##            & (lineitem_filtered.l_discount >= 0.05)
    //##            & (lineitem_filtered.l_discount <= 0.07)
    //##            & (lineitem_filtered.l_quantity < var3)
    //##        )
            %lineitem_filtered_1 = pandas.filter_date_geq(%lineitem_filtered,%date1, "l_shipdate")
            %lineitem_filtered_2 = pandas.filter_date_lt(%lineitem_filtered_1,%date2, "l_shipdate")
            %lineitem_filtered_3 = pandas.filter_num_geq(%lineitem_filtered_2,%var4, "l_discount")
            %lineitem_filtered_5 = pandas.filter_num_leq(%lineitem_filtered_4,%var5, "l_discount")
            %sel                 = pandas.filter_num_lt(%lineitem_filtered_5,%var3, "l_quantity")



    //>>  Description:
    //>>  The flineitem variable is created by filtering the lineitem_filtered dataframe using the boolean condition stored in sel.
    //>>  This filters out the rows that do not satisfy the conditions.
    //>>
    //##        flineitem = lineitem_filtered[sel]
            %flineitem         = %sel

    //>>  Description:
    //>>  calculates the result value by multiplying the "l_extendedprice" column with the "l_discount" column for each row in the flineitem
    //>>  dataframe, and then summing up the values. The result is stored in the result_value variable.
    //##        result_value = (flineitem.l_extendedprice * flineitem.l_discount).sum()
            %result_value_temp1 = pandas.addcol     (%flineitem,'l_extendedprice', 'result_value_temp')
            %result_value_temp2 = pandas.arith_mult (%flineitem,'l_discount','result_value_temp')
            %result_value       = pandas.agg_sum    (%result_value_temp2, 'result_value_temp')

    //>>  Descripttion:
    //>>  Finally, a new dataframe called result_df is created with a single column named "revenue" and the calculated result_value as its value
    //##        result_df = pd.DataFrame({"revenue": [result_value]})

    
    //##        return result_df
    //##

        llvm.return
    }
}