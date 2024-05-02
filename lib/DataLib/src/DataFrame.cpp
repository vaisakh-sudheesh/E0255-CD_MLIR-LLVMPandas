#include <iostream>
#include <fstream>
#include <sstream>
#include <sqlite3.h>
#include <vector>
#include <algorithm>
#include <functional>

#include "DataLib.hpp"

DataFrame::DataFrame(std::string name) {
    // Open an empty SQLite database in memory
    int rc = sqlite3_open(":memory:", &db);
    tableName = name;
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    // Create an empty table
    std::string createTableQuery = "CREATE TABLE "+tableName+" (rowid INTEGER PRIMARY KEY AUTOINCREMENT);";
    rc = sqlite3_exec(db, createTableQuery.c_str(), nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create table: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }
}

bool DataFrame::read_csv(const std::string& csvFilePath) {
    // Open SQLite database
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }
    std::ifstream csvFile(csvFilePath);
    std::string value__, line__;

    std::getline(csvFile, line__);
    std::istringstream iss_header(line__);
    
    // Create table to store data
    std::string createTableQuery = "CREATE TABLE " + tableName + " (";
    while (std::getline(iss_header, value__, '|')) {
        if (sqlite3_strnicmp(value__.c_str(), "column_", 7) != 0) {
            createTableQuery += "" + value__ + " TEXT,";
        }        
    }
    // Remove the trailing comma
    createTableQuery.pop_back();
    createTableQuery += ");";
    
    rc = sqlite3_exec(db, createTableQuery.c_str(), nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create table: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return false;
    }
    // std::cout << " Creation Query: "<<createTableQuery << std::endl;

    // Import data from CSV file
    int ctr = 0;
    const int max_ctr = 1000;
    std::string line;
    std::string value;
    std::string importDataQuery;
    do {
        ctr = 0;
        importDataQuery = "INSERT INTO " + tableName + " VALUES ";
        while (std::getline(csvFile, line) && (ctr < max_ctr)) {
            std::istringstream iss(line);
            std::string value;
            importDataQuery += "(";
            while (std::getline(iss, value, '|')) {
                importDataQuery += "\"" + value + "\",";
            }
            // Remove the trailing comma
            importDataQuery.pop_back();
            importDataQuery += "),";
            ctr++;
        }
        // Remove the trailing comma
        importDataQuery.pop_back();
        importDataQuery += ";";
        if (ctr) {
            // std::cout << importDataQuery << std::endl;
            rc = sqlite3_exec(db, importDataQuery.c_str(), nullptr, nullptr, nullptr);
            if (rc != SQLITE_OK) {
                std::cerr << "sql-query: " << importDataQuery << std::endl;
                std::cerr << "Failed to import data: " << sqlite3_errmsg(db) << std::endl;
                sqlite3_close(db);
                return false;
            }
            if (ctr < max_ctr) {
                break;
            }
        }
    } while (true);

    return true;
}
DataFrame::~DataFrame() {
    sqlite3_close(db);
}

bool DataFrame::hasColumn(const std::string& columnName) {
    std::string selectQuery = "SELECT * FROM " + tableName + ";";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    int numColumns = sqlite3_column_count(stmt);
    for (int i = 0; i < numColumns; i++) {
        std::string colname = reinterpret_cast<const char*>(sqlite3_column_name(stmt, i));
        if (colname == columnName) {
            sqlite3_finalize(stmt);
            return true;
        }
    }

    sqlite3_finalize(stmt);
    return false;
}

int DataFrame::getColumnSize(const std::string& columnName) {
    std::string selectQuery = "SELECT " + columnName + " FROM " + tableName + ";";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return 0;
    }

    int numRows = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        numRows++;
    }

    sqlite3_finalize(stmt);
    return numRows;
}

std::string DataFrame::getColumnValue(const std::string& columnName, int index) {
    std::string selectQuery = "SELECT " + columnName + " FROM " + tableName + " LIMIT 1 OFFSET " + std::to_string(index) + ";";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return "";
    }

    std::string value;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return value;
}

std::string DataFrame::getColumnValue(const std::string& columnName) {
    std::string selectQuery = "SELECT " + columnName + " FROM " + tableName + ";";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return "";
    }

    std::string value;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return value;
}

int DataFrame::getNumColumns () {
    std::string selectQuery = "PRAGMA table_info(" + tableName + ");";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return 0;
    }

    int numColumns = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        numColumns++;
    }

    sqlite3_finalize(stmt);
    return numColumns;
}

void DataFrame::addColumn(const std::string& columnName, const std::string& defaultValue) {
    std::string alterTableQuery = "ALTER TABLE " + tableName + " ADD " + columnName + " TEXT DEFAULT '" + defaultValue + "';";
    int rc = sqlite3_exec(db, alterTableQuery.c_str(), nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to add column: " << sqlite3_errmsg(db) << std::endl;
    }
}

void DataFrame::dumpSchemas() {
    std::string selectQuery = "PRAGMA table_info(" + tableName + ");";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* colname = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* coltype = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        const char* coldefault = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));

        std::string colnameStr = colname ? colname : "";
        std::string coltypeStr = coltype ? coltype : "";
        std::string coldefaultStr = coldefault ? coldefault : "";

        std::cout << "Column: " << colnameStr << " Type: " << coltypeStr << " Default: " << coldefaultStr << std::endl;
    }

    sqlite3_finalize(stmt);
}

void DataFrame::addData(const std::string& columnName, const std::string& value) {
    std::string insertQuery = "INSERT INTO " + tableName + " (" + columnName + ") VALUES (" + value + ");";
    std::cout<<insertQuery<<std::endl;
    int rc = sqlite3_exec(db, insertQuery.c_str(), nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to add data: " << sqlite3_errmsg(db) << std::endl;
    }
}

DataFrame DataFrame::subset(const std::vector<std::string>& columns) {
    std::string selectQuery = "SELECT ";
    for (const std::string& column : columns) {
        selectQuery += column + ",";
    }
    // Remove the trailing comma
    selectQuery.pop_back();
    selectQuery += " FROM " + tableName + ";";

    // Execute the select query and create a new DataFrame with the subset of columns
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and create a new DataFrame:
    std::cout<<selectQuery<<std::endl;
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return DataFrame();
    }

    DataFrame sDataFrame;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        // Retrieve values from the result set and construct the new DataFrame
        // You can customize this part based on your requirements
        // Here's an example of how to retrieve values and construct the new DataFrame:
        for (int i = 0; i < columns.size(); i++) {
            std::string colname = reinterpret_cast<const char*>(sqlite3_column_name(stmt, i));
            std::string value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            sDataFrame.addColumn(colname, value);
        }
    }

    sqlite3_finalize(stmt);
    return sDataFrame;
}

// 4. Row-wise operations on columns
std::vector<std::string> DataFrame::applyRowWiseOperation(const std::string& columnName, const std::function<std::string(const std::string&)>& operation) {
    std::vector<std::string> result;
    std::string selectQuery = "SELECT " + columnName + " FROM " + tableName + ";";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return result;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string newValue = operation(value);
        result.push_back(newValue);
    }

    sqlite3_finalize(stmt);
    return result;
}

// 5. Groupby - Grouping by single and multiple columns
// DataFrame groupBy(const std::vector<std::string>& groupByColumns) {
//     std::string selectQuery = "SELECT ";
//     for (const std::string& column : groupByColumns) {
//         selectQuery += column + ",";
//     }
//     // Remove the trailing comma
//     selectQuery.pop_back();
//     selectQuery += " FROM " + tableName + " GROUP BY ";
//     for (const std::string& column : groupByColumns) {
//         selectQuery += column + ",";
//     }
//     // Remove the trailing comma
//     selectQuery.pop_back();
//     selectQuery += ";";

//     // Execute the select query and create a new DataFrame with the grouped columns
//     sqlite3_stmt* stmt;
//     int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
//     if (rc != SQLITE_OK) {
//         std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
//         return DataFrame();
//     }

//     DataFrame groupedDataFrame;
//     while (sqlite3_step(stmt) == SQLITE_ROW) {
//         // Retrieve values from the result set and construct the new DataFrame
//         // You can customize this part based on your requirements
//         // Here's an example of how to retrieve values and construct the new DataFrame:
//         std::vector<std::string> groupValues;
//         for (int i = 0; i < groupByColumns.size(); i++) {
//             std::string value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
//             groupValues.push_back(value);
//         }

//         // Add the retrieved values to the new DataFrame
//         // You can customize this part based on your requirements
//         // Here's an example of how to add values to the new DataFrame:
//         for (const std::string& column : groupByColumns) {
//             groupedDataFrame.addColumn(column, groupValues[i]);
//             // Remove the trailing comma
//             selectQuery.pop_back();
//             selectQuery += " FROM " + tableName + ";";

//             // Execute the select query and create a new DataFrame with the subset of columns
//             // You can customize this part based on your requirements
//             // Here's an example of how to execute the query and create a new DataFrame:
//             sqlite3_stmt* stmt;
//             int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
//             if (rc != SQLITE_OK) {
//                 std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
//                 return DataFrame();
//             }

//             DataFrame subsetDataFrame;
//             while (sqlite3_step(stmt) == SQLITE_ROW) {
//                 // Retrieve values from the result set and construct the new DataFrame
//                 // You can customize this part based on your requirements
//                 // Here's an example of how to retrieve values and construct the new DataFrame:
//                 std::string column1 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
//                 std::string column2 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
//                 std::string column3 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

//                 // Add the retrieved values to the new DataFrame
//                 // You can customize this part based on your requirements
//                 // Here's an example of how to add values to the new DataFrame:
//                 subsetDataFrame.addColumn("column1", column1);
//                 subsetDataFrame.addColumn("column2", column2);
//                 subsetDataFrame.addColumn("column3", column3);
//             }

//             sqlite3_finalize(stmt);
//             return subsetDataFrame;
//         }
//     }
// }

// Implement the remaining operations (row-wise operations, groupby, filtering, reductions, join, sorting) based on your requirements
// 6. Filtering - Only simple predicates on columns (like <, > etc). No UDFs.
DataFrame DataFrame::filter(const std::string& columnName, const std::string& predicate, const std::string& value) {
    std::string selectQuery = "SELECT * FROM " + tableName + " WHERE " + columnName + " " + predicate + " " + value + ";";
    // Execute the select query and create a new DataFrame with the filtered rows
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and create a new DataFrame:
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return DataFrame();
    }

    DataFrame filteredDataFrame;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        // Retrieve values from the result set and construct the new DataFrame
        // You can customize this part based on your requirements
        // Here's an example of how to retrieve values and construct the new DataFrame:
        for (int i = 0; i < columnName.size(); i++) {
            std::string colname = reinterpret_cast<const char*>(sqlite3_column_name(stmt, i));
            std::string value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            filteredDataFrame.addColumn(colname, value);
        }
    }

    sqlite3_finalize(stmt);
    return filteredDataFrame;
}

// 7. Reductions - Sum, Mean, Count
std::string DataFrame::sum(const std::string& columnName) {
    std::string selectQuery = "SELECT SUM(" + columnName + ") FROM " + tableName + ";";
    // Execute the select query and retrieve the sum
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and retrieve the sum:
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return "";
    }

    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return result;
}

std::string DataFrame::mean(const std::string& columnName) {
    std::string selectQuery = "SELECT AVG(" + columnName + ") FROM " + tableName + ";";
    // Execute the select query and retrieve the mean
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and retrieve the mean:
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return "";
    }

    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return result;
}

std::string DataFrame::count(const std::string& columnName) {
    std::string selectQuery = "SELECT COUNT(" + columnName + ") FROM " + tableName + ";";
    // Execute the select query and retrieve the count
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and retrieve the count:
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return "";
    }

    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return result;
}

// 8. Join: Only equi-joins where columns are explicitly specified
DataFrame DataFrame::join(const DataFrame& otherDataFrame, const std::string& column1, const std::string& column2) {
    std::string selectQuery = "SELECT * FROM " + tableName + " INNER JOIN " + otherDataFrame.tableName + " ON " + tableName + "." + column1 + " = " + otherDataFrame.tableName + "." + column2 + ";";
    // Execute the select query and create a new DataFrame with the joined rows
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and create a new DataFrame:
    std::cout<<selectQuery<<std::endl;
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return DataFrame();
    }

    DataFrame joinedDataFrame;
    int numColumns = sqlite3_column_count(stmt);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        // Retrieve values from the result set and construct the new DataFrame
        // You can customize this part based on your requirements
        // Here's an example of how to retrieve values and construct the new DataFrame:
        for (int i = 0; i < numColumns; i++) {
            std::string columnName = reinterpret_cast<const char*>(sqlite3_column_name(stmt, i));
            std::string columnValue = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            joinedDataFrame.addColumn(columnName, columnValue);
        }
    }

    sqlite3_finalize(stmt);
    return joinedDataFrame;
}

// 9. Sorting by a column (ORDER BY)
DataFrame DataFrame::sortByColumn(const std::string& columnName) {
    std::string selectQuery = "SELECT * FROM " + tableName + " ORDER BY " + columnName + ";";
    // Execute the select query and create a new DataFrame with the sorted rows
    // You can customize this part based on your requirements
    // Here's an example of how to execute the query and create a new DataFrame:
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, selectQuery.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to execute select query: " << sqlite3_errmsg(db) << std::endl;
        return DataFrame();
    }

    DataFrame sortedDataFrame;
    int numColumns = sqlite3_column_count(stmt);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        // Retrieve values from the result set and construct the new DataFrame
        // You can customize this part based on your requirements
        // Here's an example of how to retrieve values and construct the new DataFrame:
        for (int i = 0; i < numColumns; i++) {
            std::string columnName = reinterpret_cast<const char*>(sqlite3_column_name(stmt, i));
            std::string columnValue = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            sortedDataFrame.addColumn(columnName, columnValue);
        }
    }

    sqlite3_finalize(stmt);
    return sortedDataFrame;
}


std::string DataFrame::getTableName() {
    return tableName;
}
