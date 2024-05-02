#include "gtest/gtest.h"
#include "DataLib.hpp" // Include your library

#include <string>
std::string testdata_dir = "/home/vaisakh/developer/Compiler/Project/github/E0_255-CD_MLIR-Project/tpc-benchmarks/tables_scale_1/" ;


////////////// Read CSV test sets - BEGIN //////////////
/**
 * Utility function to load data from a CSV fileand test the read_csv() function
 */
void test_read_csv(std::string filename) {
}

class ReadCSVTest : public testing::TestWithParam<const char *> {};

TEST_P(ReadCSVTest, tests) {
    DataFrame *dataLib = new DataFrame();
    ASSERT_EQ(dataLib->read_csv(testdata_dir+GetParam()), true);
    delete dataLib;
}
INSTANTIATE_TEST_SUITE_P(
    DataLibTest,
    ReadCSVTest,
    ::testing::Values(
        "customer.csv",
        "lineitem.csv",
        "nation.csv",
        "orders.csv",
        "part.csv",
        "partsupp.csv",
        "region.csv",
        "supplier.csv"
    )
);

////////////// Read CSV test sets - END //////////////

TEST(DataFrameTest, AddColumnTest) {
  DataFrame df;
  std::string columnName = "Age";
  std::string defaultValue = "25";

  df.addColumn(columnName, defaultValue);
  df.dumpSchemas();

  df.addData("Age", "25");

  // Assert that the column was added successfully
  ASSERT_TRUE(df.hasColumn(columnName));
  ASSERT_EQ(df.getColumnSize(columnName), 1);
  ASSERT_EQ(df.getColumnValue(columnName, 0), defaultValue);
}

TEST(DataFrameTest, SubsetTest) {
    // Create a DataFrame object
    DataFrame df;
    // Add some columns to the DataFrame
    df.addColumn("NAME", "John");
    df.addColumn("AGE", "25");
    df.addColumn("CITY", "New York");
    df.dumpSchemas();

    df.addData("NAME,AGE,CITY", "'John','25','New'");

    // Define the columns to subset
    std::vector<std::string> columns = {"NAME", "CITY"};

    // Call the subset function
    DataFrame subset = df.subset(columns);

    // Assert that the subset DataFrame has the correct number of columns
    EXPECT_EQ(subset.getNumColumns(), columns.size());

    // Assert that the subset DataFrame has the correct column names and values
    EXPECT_EQ(subset.getColumnValue("NAME"), "John");
    EXPECT_EQ(subset.getColumnValue("CITY"), "New York");
}

TEST(DataFrameTest, ApplyRowWiseOperation) {
    // Create a DataFrame object
    DataFrame df;

    df.addColumn("column1", "values");
    df.dumpSchemas();
    // Add some data to the DataFrame
    df.addData("column1", "'value1'");
    df.addData("column1", "'value2'");
    df.addData("column1", "'value3'");

    // Define the operation to be applied row-wise
    std::function<std::string(const std::string&)> operation = [](const std::string& value) {
        return value + "_modified";
    };

    // Apply the row-wise operation
    std::vector<std::string> result = df.applyRowWiseOperation("column1", operation);

    // Verify the result
    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "value1_modified");
    EXPECT_EQ(result[1], "value2_modified");
    EXPECT_EQ(result[2], "value3_modified");
}


// Test case for sum function
TEST(DataFrameTest, SumTest) {
    // Create a DataFrame object
    DataFrame df;

    df.addColumn("column1", "values");
    df.dumpSchemas();
    df.addData("column1", "10");
    df.addData("column1", "20");

    // Call the sum function with a column name
    std::string columnName = "column1";
    std::string sumResult = df.sum(columnName);

    // Assert the expected sum result
    std::string expectedSumResult = "30"; // Replace with your expected sum result
    EXPECT_EQ(sumResult, expectedSumResult);
}
TEST(DataFrameTest, JoinTest) {
  // Create two data frames
  DataFrame df1("table1");
  df1.addColumn("id", "1");
  df1.addColumn("name", "John");

  DataFrame df2("table2");
  df2.addColumn("id", "1");
  df2.addColumn("age", "25");

  // Perform join operation
  DataFrame joined = df1.join(df2, "id", "id");

  // Verify the joined data frame
  EXPECT_EQ(joined.getTableName(), "table1");
  EXPECT_EQ(joined.getNumColumns(), 4); // Assuming the join adds two columns

  // Verify the joined rows
  EXPECT_EQ(joined.getColumnValue("id", 0), "1");
  EXPECT_EQ(joined.getColumnValue("name", 0), "John");
  EXPECT_EQ(joined.getColumnValue("id", 1), "1");
  EXPECT_EQ(joined.getColumnValue("age", 1), "25");
}

TEST(DataFrameTest, JoinTest_EmptyDataFrames) {
  // Create two empty data frames
  DataFrame df1;
  DataFrame df2;

  // Perform join operation
  DataFrame joined = df1.join(df2, "id", "id");

  // Verify the joined data frame
  EXPECT_EQ(joined.getTableName(), "");
  EXPECT_EQ(joined.getNumColumns(), 0);
}

TEST(DataFrameTest, JoinTest_NoMatchingRows) {
  // Create two data frames with no matching rows
  DataFrame df1("table1");
  df1.addColumn("id", "1");
  df1.addColumn("name", "John");

  DataFrame df2("table2");
  df2.addColumn("id", "2");
  df2.addColumn("age", "25");

  // Perform join operation
  DataFrame joined = df1.join(df2, "id", "id");

  // Verify the joined data frame
  EXPECT_EQ(joined.getTableName(), "table1");
  EXPECT_EQ(joined.getNumColumns(), 2); // Assuming no new columns are added

  // Verify the joined rows
  EXPECT_EQ(joined.getColumnValue("id", 0), "1");
  EXPECT_EQ(joined.getColumnValue("name", 0), "John");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}