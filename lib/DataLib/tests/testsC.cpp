#include <iostream>
#include "gtest/gtest.h"

#include "DataLib.h"

#ifdef USE_TPCH_DATA
const char *tpch_testdata_dir = "/home/vaisakhps/developer/CDProject-v2/tpc-benchmarks/tables_scale_1/";
#define TESTDATASET "customer.csv","lineitem.csv","nation.csv","orders.csv","part.csv","partsupp.csv","region.csv","supplier.csv"
#else
const char *tpch_testdata_dir = "/home/vaisakhps/developer/CDProject-v2/lib/DataLib/tests/";
#define TESTDATASET "test_data.csv"
#endif // USE_TPCH_DATA

extern "C" void printTableNamesAndSchema();

////////////// Read CSV test sets - BEGIN //////////////
class ReadCSVTest : public testing::TestWithParam<const char *> {};

TEST_P(ReadCSVTest, passtest) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    DataFrame_free(dataLib);
}

TEST_P(ReadCSVTest, nofile) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    strcat(filename, "no_file.csv");
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib == nullptr);
    DataFrame_free(dataLib);
}

TEST_P(ReadCSVTest, invalid_input1) {
    DataFrame_deleteAllTables();
    char filename[1024];
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),nullptr, 10);
    ASSERT_TRUE(dataLib == nullptr);
    DataFrame_free(dataLib);
}

TEST_P(ReadCSVTest, invalid_input2) {
    DataFrame_deleteAllTables();
    char filename[1024];
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, -1);
    ASSERT_TRUE(dataLib == nullptr);
    DataFrame_free(dataLib);
}

TEST_P(ReadCSVTest, invalid_input3) {
    DataFrame_deleteAllTables();
    char filename[1024];
    void *dataLib = DataFrame_read_csv(nullptr, strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib == nullptr);
    DataFrame_free(dataLib);
}

INSTANTIATE_TEST_SUITE_P(DataLibTest, ReadCSVTest,::testing::Values(TESTDATASET));
////////////// Read CSV test sets - END //////////////

////////////// Aggregation test sets - BEGIN //////////////
class AggregationTest : public testing::TestWithParam<const char *> {};
TEST_P(AggregationTest, Sum) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    float result = DataFrame_sum(dataLib, "c_acctbal", strlen("c_acctbal"));
    EXPECT_FLOAT_EQ(result, 139376.43f);
    DataFrame_free(dataLib);
}
TEST_P(AggregationTest, Min) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    float result = DataFrame_min(dataLib, "c_acctbal", strlen("c_acctbal"));
    EXPECT_FLOAT_EQ(result, -272.6f);
    DataFrame_free(dataLib);
}

TEST_P(AggregationTest, Max) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    float result = DataFrame_max(dataLib, "c_acctbal", strlen("c_acctbal"));
    EXPECT_FLOAT_EQ(result, 9561.95f);
    DataFrame_free(dataLib);
}

TEST_P(AggregationTest, Mean) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    float result = DataFrame_avg(dataLib, "c_acctbal", strlen("c_acctbal"));
    EXPECT_FLOAT_EQ(result, 4645.881);
    DataFrame_free(dataLib);
}

TEST_P(AggregationTest, Count) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    int result = DataFrame_count(dataLib, "c_acctbal", strlen("c_acctbal"));
    EXPECT_EQ(result, 30);
    DataFrame_free(dataLib);
}

INSTANTIATE_TEST_SUITE_P(DataLibTest, AggregationTest,::testing::Values(TESTDATASET));

////////////// Aggregation test sets - END //////////////

////////////// Column test sets - BEGIN //////////////
class ColumnTests : public testing::TestWithParam<const char *> {};
TEST_P(ColumnTests, ColumnExists) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    int result = DataFrame_hasColumn(dataLib, "c_acctbal", strlen("c_acctbal"));
    EXPECT_EQ(result, 1);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, ColumnNotExists) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    int result = DataFrame_hasColumn(dataLib, "c_acctbal1", strlen("c_acctbal1"));
    EXPECT_EQ(result, 0);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, InvalidInput1) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    int result = DataFrame_hasColumn(dataLib, nullptr, 0);
    EXPECT_EQ(result, 0);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, InvalidInput2) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    int result = DataFrame_hasColumn(dataLib, nullptr, 10);
    EXPECT_EQ(result, 0);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, PrintRows) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    DataFrame_printRows(dataLib, 5);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, SubsetData) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    // std::cout<< "=====Original DF (test_table)=====" << std::endl;
    // DataFrame_printRows(dataLib, 5);
    void *newdataLib = DataFrame_subset(dataLib, "c_acctbal,c_name", strlen("c_acctbal,c_name"), "subsettable", strlen("subsettable"));
    ASSERT_TRUE(newdataLib != nullptr);
    // std::cout<< "=====Original DF (new_table)=====" << std::endl;
    // DataFrame_printRows(newdataLib, 5);
    DataFrame_free(newdataLib);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, MergeTest) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    std::cout<< "=====Table before subset creation =====" << std::endl;
    printTableNamesAndSchema();
    ASSERT_TRUE(dataLib != nullptr);
    void *newdataLib1 = DataFrame_subset(dataLib, "c_acctbal,c_name", strlen("c_acctbal,c_name"), "subsettable1", strlen("subsettable1"));
    ASSERT_TRUE(newdataLib1 != nullptr);
    std::cout<< "=====Subset Table1 =====" << std::endl;
    DataFrame_printRows(newdataLib1, 5);
    void *newdataLib2 = DataFrame_subset(dataLib, "c_phone,c_nationkey", strlen("c_phone,c_nationkey"), "subsettable2", strlen("subsettable2"));
    ASSERT_TRUE(newdataLib2 != nullptr);
    std::cout<< "=====Subset Table2 =====" << std::endl;
    DataFrame_printRows(newdataLib2, 5);
    
    std::cout<< "=====Tables/Schema after subset creation =====" << std::endl;
    printTableNamesAndSchema();

    void * mergeddf =  DataFrame_merge(newdataLib1, newdataLib2, "merged_table", strlen("merged_table"));
    ASSERT_TRUE(mergeddf != nullptr);
    std::cout<< "=====Tables/Schema after merge =====" << std::endl;
    printTableNamesAndSchema();
    std::cout<< "=====Merged DF (mergeddf)=====" << std::endl;
    DataFrame_printRows(mergeddf, 5);


    DataFrame_free(mergeddf);
    DataFrame_free(newdataLib2);
    DataFrame_free(newdataLib1);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, GroupbyTest) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *groupedDF = DataFrame_groupby(dataLib, "c_nationkey", strlen("c_nationkey"), "grouped_table", strlen("grouped_table"),0);
    ASSERT_TRUE(groupedDF != nullptr);
    std::cout<< "=====Tables/Schema before GroupBy =====" << std::endl;
    printTableNamesAndSchema();
    std::cout<< "=====Grouped DF (groupedDF)=====" << std::endl;
    DataFrame_printRows(groupedDF, 30);
}

TEST_P(ColumnTests, GroupbyTest2) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *groupedDF = DataFrame_groupby(dataLib, "c_nationkey,c_phone", strlen("c_nationkey,c_phone"), "grouped_table", strlen("grouped_table"),1);
    ASSERT_TRUE(groupedDF != nullptr);
    std::cout<< "=====Tables/Schema before GroupBy =====" << std::endl;
    printTableNamesAndSchema();
    std::cout<< "=====Grouped DF (groupedDF)=====" << std::endl;
    DataFrame_printRows(groupedDF, 30);
}


TEST_P(ColumnTests, ArithAdditionColumns) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_AritheMeticOperation_Add(dataLib, "c_acctbal", strlen("c_acctbal"), "c_acctbal", strlen("c_acctbal"));
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====ArithAdditionColumns DF =====" << std::endl;
    DataFrame_printRows(dataLib, 5);
}

TEST_P(ColumnTests, ArithAdditionScalar) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_AritheMeticOperation_ScalarAdd(dataLib, "c_acctbal", strlen("c_acctbal"), 200);
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====ArithAdditionScalar DF =====" << std::endl;
    DataFrame_printRows(dataLib, 5);
}

TEST_P(ColumnTests, ArithSubtractionColumns) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_AritheMeticOperation_Subtract(dataLib, "c_nationkey", strlen("c_nationkey"), "c_acctbal", strlen("c_acctbal"));
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====ArithSubtractionColumns DF =====" << std::endl;
    DataFrame_printRows(dataLib, 5);
}

TEST_P(ColumnTests, ArithSubtractionScalar) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_AritheMeticOperation_ScalarSubtract(dataLib, "c_acctbal", strlen("c_acctbal"), 100);
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====ArithSubtractionScalar DF =====" << std::endl;
    DataFrame_printRows(dataLib, 5);
}
    
TEST_P(ColumnTests, ArithMultiplicationColumns) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_AritheMeticOperation_Multiply(dataLib, "c_nationkey", strlen("c_nationkey"), "c_acctbal", strlen("c_acctbal"));
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====ArithMultiplicationColumns DF =====" << std::endl;
    DataFrame_printRows(dataLib, 5);
}

TEST_P(ColumnTests, ArithMultiplicationScalar) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_AritheMeticOperation_ScalarMultiply(dataLib, "c_acctbal", strlen("c_acctbal"), 100);
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====ArithMultiplicationScalar DF =====" << std::endl;
    DataFrame_printRows(dataLib, 5);
}

TEST_P(ColumnTests, SortRowsAscending) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    std::cout<< "=====Before sorting printing rows=====" << std::endl;
    DataFrame_printRows(dataLib, 5);
    void *newdataLib = DataFrame_sortByColumn(dataLib, "c_acctbal", strlen("c_acctbal"), 0, "new_table", strlen("new_table"));
    std::cout<< "=====After sorting printing rows=====" << std::endl;
    DataFrame_printRows(newdataLib, 5);
    std::cout<< "=====Print test complete=====" << std::endl;
    DataFrame_free(newdataLib);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, SortRowsDescending) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    ASSERT_TRUE(dataLib != nullptr);
    std::cout<< "=====Before sorting printing rows=====" << std::endl;
    DataFrame_printRows(dataLib, 5);
    void *newdataLib = DataFrame_sortByColumn(dataLib, "c_acctbal", strlen("c_acctbal"), 1, "new_table", strlen("new_table"));
    std::cout<< "=====After sorting printing rows=====" << std::endl;
    DataFrame_printRows(newdataLib, 5);
    std::cout<< "=====Print test complete=====" << std::endl;
    DataFrame_free(newdataLib);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, AddColumnTest) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib1 = DataFrame_read_csv("test_table1", strlen("test_table1"),filename, strlen(filename));
    void *res = DataFrame_addColumn(dataLib1, "c_acctbalnew", strlen("c_acctbalnew"), "c_acctbal", strlen("c_acctbal"));
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====AddColumnTest DF =====" << std::endl;
    DataFrame_printRows(dataLib1, 5);
    DataFrame_free(dataLib1);
}

TEST_P(ColumnTests, FilterTest1) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_filter(dataLib, 
                                "c_acctbal", strlen("c_acctbal"),
                                "<", strlen("<"),
                                "800", strlen("800"),
                                "filtered_table", strlen("filtered_table"));
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====FilterTest1 DF =====" << std::endl;
    DataFrame_printRows(res, 5);

    DataFrame_free(res);
    DataFrame_free(dataLib);
}

TEST_P(ColumnTests, FilterTest2) {
    DataFrame_deleteAllTables();
    char filename[1024];
    strcpy(filename, tpch_testdata_dir);
    strcat(filename, GetParam());
    void *dataLib = DataFrame_read_csv("test_table", strlen("test_table"),filename, strlen(filename));
    void *res = DataFrame_filter(dataLib, 
                                "c_acctbal", strlen("c_acctbal"),
                                ">", strlen(">"),
                                "800", strlen("800"),
                                "filtered_table", strlen("filtered_table"));
    ASSERT_TRUE(res != nullptr);
    std::cout<< "=====FilterTest1 DF =====" << std::endl;
    DataFrame_printRows(res, 5);

    DataFrame_free(res);
    DataFrame_free(dataLib);
}



INSTANTIATE_TEST_SUITE_P(DataLibTest, ColumnTests,::testing::Values(TESTDATASET));
////////////// Column test sets - END //////////////


int main(int argc, char **argv) {
    DataFrame_deleteAllTables();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}