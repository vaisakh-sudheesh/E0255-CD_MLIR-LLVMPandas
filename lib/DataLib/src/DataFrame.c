#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>
#include <string.h>

#include "DataLib.h"
#define MAX_TABLES 10


static sqlite3* db = NULL;

typedef struct {    
    
    char* tableName;
} DataFrame;


/// @brief Replace CSV delimiter with SQL delimiter
/// @param line 
/// @note This function is not exposed to the user. It is used internally by DataFrame_read_csv.
static void __replace_csv_delimiter(char* line) {
    if (line == NULL) {
        return;
    }
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == '|') {
            line[i] = ',';
        }
    }
}

/// @brief Special case handling for the last CSV separator in a line
/// @param str 
/// @note This function is not exposed to the user. It is used internally by DataFrame_read_csv.
static void remove_after_last_delim(char* str, char delim) {
    if (str == NULL) {
        return;
    }
    char* lastComma = strrchr(str, delim);
    if (lastComma != NULL) {
        *lastComma = '\0';
    }
}

/// @brief Strip off existing separators and add new ones compatible with SQL
/// @param sqlValues 
/// @param line 
/// @note This function is not exposed to the user. It is used internally by DataFrame_read_csv.
static void __form_sql_values(char *sqlValues, char *line) {
    if (line == NULL || sqlValues == NULL) {
        return;
    }
    char *token = strtok(line, "|");
    sqlValues[0] = '\0';
    while (token != NULL) {
        strcat(sqlValues, "\"");
        strcat(sqlValues, token);
        strcat(sqlValues, "\"|");
        token = strtok(NULL, "|");
    }
    remove_after_last_delim(sqlValues,'|');
}

/// @brief Delete all tables from SQLite database and perform VACUUM and INTEGRITY check
/// @param df Pointer to the DataFrame instance
void DataFrame_deleteAllTables() {
    if (db == NULL) {
        int rc = sqlite3_open(":memory:", &db);
        if (rc != SQLITE_OK) {
            return;
        }
    }
    
    char deleteTablesQuery[256];
    sprintf(deleteTablesQuery, "SELECT name FROM sqlite_master WHERE type='table';");
    
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, deleteTablesQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(db);
        db = NULL;
        return;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* tableName = sqlite3_column_text(stmt, 0);
        char dropTableQuery[256];
        sprintf(dropTableQuery, "DROP TABLE IF EXISTS %s;", tableName);
        rc = sqlite3_exec(db, dropTableQuery, NULL, NULL, NULL);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            db = NULL;
            return;
        }
    }
    
    sqlite3_finalize(stmt);
    
    // Perform VACUUM
    rc = sqlite3_exec(db, "VACUUM;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(db);
        db = NULL;
        return;
    }
    
    // Perform INTEGRITY check
    rc = sqlite3_exec(db, "PRAGMA integrity_check;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        return;
    }

    sqlite3_close(db);
    db = NULL;
}

/// @brief Internal function to create a new DataFrame instance. 
/// @param name 
/// @return Pointer to the new DataFrame instance. 
/// @note This function is not exposed to the user. An instance of DataFrame is created using DataFrame_read_csv.
/// @note The user should call DataFrame_free to free the memory allocated for the DataFrame instance.
static void* DataFrame_new(const char* name, const char* columns) {
    if (name == NULL) {
        return NULL;
    }
    DataFrame* df = malloc(sizeof(DataFrame));
    df->tableName = strdup(name);
    if (db == NULL) {
        int rc = sqlite3_open(":memory:", &db);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Failed to open database: %s\n", sqlite3_errmsg(db));
            return NULL;
        }
    }

    char createTableQuery[256];
    sprintf(createTableQuery, "CREATE TABLE %s (rowid INTEGER PRIMARY KEY AUTOINCREMENT%s%s);", 
                df->tableName,
                ((columns != NULL) ? "," : ""),
                ((columns != NULL) ? columns : "")
            );
    
    int rc = sqlite3_exec(db, createTableQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_new:: Failed to create table: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df;
}

static void* DataFrame_new_NoTabCreate(const char* name) {
    if (name == NULL) {
        return NULL;
    }
    DataFrame* df = malloc(sizeof(DataFrame));
    df->tableName = strdup(name);
    if (db == NULL) {
        int rc = sqlite3_open(":memory:", &db);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "Failed to open database: %s\n", sqlite3_errmsg(db));
            return NULL;
        }
    }
    return df;
}

/// @brief Free the memory allocated for the DataFrame instance
/// @param df 
/// @note This function is not exposed to the user. The user should call this function to free the memory allocated for the DataFrame instance.
void DataFrame_free(void* df) {
    DataFrame *df_ = (DataFrame*)df;
    if (df_ != NULL) {
        free(df_->tableName);
        free(df_);
    }
}

/// @brief Read a CSV file into a DataFrame
/// @param tableName 
/// @param sztableName 
/// @param csvFilePath 
/// @param szcsvFilePath 
/// @return Pointer to the DataFrame instance
/// @note The user should call DataFrame_free to free the memory allocated for the DataFrame instance.
void* DataFrame_read_csv(const char* tableName, int sztableName, const char* csvFilePath, int szcsvFilePath){
    if (tableName == NULL || csvFilePath == NULL || sztableName <= 0 || szcsvFilePath <= 0) {
        return NULL;
    }
    char tableName_[sztableName+1] , csvFilePath_[szcsvFilePath+1];
    memset(tableName_, 0, sztableName+1);
    memset(csvFilePath_, 0, szcsvFilePath+1);
    strncpy(tableName_, tableName, sztableName);
    strncpy(csvFilePath_, csvFilePath, szcsvFilePath);

    
    // fprintf(stderr, "DataFrame_read_csv:: tableName(%d): %s \n", sztableName, tableName);
    // fprintf(stderr, "DataFrame_read_csv:: csvFilePath(%d): %s\n", szcsvFilePath, csvFilePath_);

    if (db == NULL) {
        int rc = sqlite3_open(":memory:", &db);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "DataFrame_read_csv:: Failed to open database: %s\n", sqlite3_errmsg(db));
            return NULL;
        }
    }
    // Rest of the read_csv implementation goes here
    FILE *csvFile = fopen(csvFilePath_, "r");
    if (csvFile == NULL) {
        fprintf(stderr, "DataFrame_read_csv:: Failed to open file: %s\n", csvFilePath_);
        return NULL;
    }
   char colnames_insert[1024];
    if (fgets(colnames_insert, 1024, csvFile) == NULL) {
        fprintf(stderr, "DataFrame_read_csv:: Failed to read header from file: %s\n", csvFilePath_);
        return NULL;
    }
    remove_after_last_delim(colnames_insert,'|');
    __replace_csv_delimiter(colnames_insert);

    char colnames[1024];
    char colinfoFilePath[256];
    snprintf(colinfoFilePath, sizeof(colinfoFilePath), "%s-sqlcolinfo", csvFilePath_);
    FILE *colinfoFile = fopen(colinfoFilePath, "r");
    if (colinfoFile == NULL) {
        fprintf(stderr, "DataFrame_read_csv:: Failed to open column info file: %s\n", colinfoFilePath);
        return NULL;
    }
    if (fgets(colnames, 1024, colinfoFile) == NULL) {
        fprintf(stderr, "DataFrame_read_csv:: Failed to read column names from file: %s\n", colinfoFilePath);
        fclose(colinfoFile);
        return NULL;
    }
    fclose(colinfoFile);

    DataFrame *df_ = DataFrame_new(tableName_, colnames);
    int rc = 0;
    char line[1024];
    while (fgets(line, 1024, csvFile) != NULL) {
        char insertRowQuery[1024];
        char insertRowQuery_Values[1024];
        remove_after_last_delim(line,'|');
        __form_sql_values(insertRowQuery_Values, line);
        __replace_csv_delimiter(insertRowQuery_Values);


        sprintf(insertRowQuery, "INSERT INTO %s (%s) VALUES (%s);", df_->tableName, colnames_insert, insertRowQuery_Values);
        rc = sqlite3_exec(db, insertRowQuery, NULL, NULL, NULL);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "DataFrame_read_csv:: Failed to insert row: %s %s\n", sqlite3_errmsg(db), insertRowQuery);
            return 0;
        }
    }
    fclose(csvFile);
    
    return df_;
}

/// @brief Sum of a column
/// @param df 
/// @param columnName 
/// @param szcolumnName 
/// @return Floating point value with summation of the column
float DataFrame_sum (void* df, const char* columnName, int szcolumnName) {
    if (df == NULL || columnName == NULL || szcolumnName <= 0 ) {
        return 0.0;
    }
    char columnName_[szcolumnName+1];
    memset(columnName_, 0, szcolumnName+1);
    strncpy(columnName_, columnName, szcolumnName);

    DataFrame *df_ = (DataFrame*)df;
    char sumQuery[1024];

    sprintf(sumQuery, "SELECT SUM(%s) FROM %s;", columnName_, df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sumQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare sum query: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Failed to get sum: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    char *sum = strdup((char*)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return atof(sum);
}

/// @brief Average of a column
/// @param df 
/// @param columnName_ 
/// @param szcolumnName 
/// @return Floating point value with average of the column
float DataFrame_avg (void* df, const char* columnName_, int szcolumnName) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 ) {
        return 0.0;
    }
    char columnName[szcolumnName+1];
    memset(columnName, 0, szcolumnName+1);
    strncpy(columnName, columnName_, szcolumnName);

    DataFrame *df_ = (DataFrame*)df;
    char avgQuery[1024];
    sprintf(avgQuery, "SELECT AVG(%s) FROM %s;", columnName, df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, avgQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare avg query: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Failed to get avg: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    char *avg = strdup((char*)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return atof(avg);
}

/// @brief Min value of a column
/// @param df 
/// @param columnName_ 
/// @param szcolumnName 
/// @return Floating point value with minimum value of the column
float DataFrame_min (void* df, const char* columnName_, int szcolumnName) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 ) {
        return 0.0;
    }
    char columnName[szcolumnName+1];
    memset(columnName, 0, szcolumnName+1);
    strncpy(columnName, columnName_, szcolumnName);

    DataFrame *df_ = (DataFrame*)df;
    char minQuery[1024];
    sprintf(minQuery, "SELECT MIN(%s) FROM %s;", columnName, df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, minQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare min query: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Failed to get min: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    char *min = strdup((char*)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return atof(min);
}

/// @brief Max value of a column 
/// @param df 
/// @param columnName_ 
/// @param szcolumnName 
/// @return Floating point value with maximum value of the column
float DataFrame_max (void* df, const char* columnName_, int szcolumnName) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 ) {
        return 0.0;
    }
    char columnName[szcolumnName+1];
    memset(columnName, 0, szcolumnName+1);
    strncpy(columnName, columnName_, szcolumnName);
    
    DataFrame *df_ = (DataFrame*)df;
    char maxQuery[1024];
    sprintf(maxQuery, "SELECT MAX(%s) FROM %s;", columnName, df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, maxQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare max query: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Failed to get max: %s\n", sqlite3_errmsg(db));
        return 0.0f;
    }
    char *max = strdup((char*)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return atof(max);
}

/// @brief Count of a column
/// @param df 
/// @param columnName_ 
/// @param szcolumnName 
/// @return Integer value with count of the column
int DataFrame_count (void* df, const char* columnName_, int szcolumnName) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 ) {
        return 0.0;
    }
    char columnName[szcolumnName+1];
    memset(columnName, 0, szcolumnName+1);
    strncpy(columnName, columnName_, szcolumnName);
    
    DataFrame *df_ = (DataFrame*)df;
    char countQuery[1024];
    sprintf(countQuery, "SELECT COUNT(%s) FROM %s;", columnName, df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, countQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare count query: %s\n", sqlite3_errmsg(db));
        return 0;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Failed to get count: %s\n", sqlite3_errmsg(db));
        return 0;
    }
    char *count = strdup((char*)sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return atoi(count);
}

/// @brief Check if a column exists in the DataFrame
/// @param df 
/// @param columnName_ 
/// @param szcolumnName 
/// @return 0 if the column does not exist, 1 if it does
int DataFrame_hasColumn(void* df, const char* columnName_, int szcolumnName) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 ) {
        return 0;
    }
    char columnName[szcolumnName+1];
    memset(columnName, 0, szcolumnName+1);
    strncpy(columnName, columnName_, szcolumnName);

    DataFrame *df_ = (DataFrame*)df;
    char hasColumnQuery[1024];
    sprintf(hasColumnQuery, "PRAGMA table_info(%s);", df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, hasColumnQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare hasColumn query: %s\n", sqlite3_errmsg(db));
        return 0;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        if (strcmp(columnName, (char*)sqlite3_column_text(stmt, 1)) == 0) {
            sqlite3_finalize(stmt);
            return 1;
        }
    }
    sqlite3_finalize(stmt);
    return 0;
}

/// TODO: FIX AND VALIDATE Sorting by column functionality

/// @brief Sort the DataFrame by a column
/// @param df 
/// @param columnName_ 
/// @param szcolumnName 
/// @return 0 if the column does not exist, 1 if it does
void* DataFrame_sortByColumn(void* df, const char* columnName_, int szcolumnName, int ascending , const char* newtablename_, int sznewtablename) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 || newtablename_ == NULL || sznewtablename <= 0) {
        return NULL;
    }
    char columnName[szcolumnName+1] , newtablename[sznewtablename+1];
    memset(columnName, 0, szcolumnName+1);
    memset(newtablename, 0, sznewtablename+1);
    strncpy(columnName, columnName_, szcolumnName);
    strncpy(newtablename, newtablename_, sznewtablename);

    DataFrame *df_ = (DataFrame*)df;
    int rc;
    // char sortQuery[1024];
    // sprintf(sortQuery, "SELECT * FROM %s ORDER BY %s %s;", df_->tableName, columnName, ascending ? "ASC" : "DESC");
    // sqlite3_stmt *stmt;
    // int rc = sqlite3_prepare_v2(db, sortQuery, -1, &stmt, NULL);
    // if (rc != SQLITE_OK) {
    //     fprintf(stderr, "Failed to prepare sort query: %s\n", sqlite3_errmsg(db));
    //     return NULL;
    // }

    DataFrame *newDf = DataFrame_new_NoTabCreate(newtablename);
    char createTableQuery[1024];
    sprintf(createTableQuery, "CREATE TABLE %s AS SELECT * FROM %s ORDER BY %s %s;", newDf->tableName, df_->tableName, columnName, ascending ? "ASC" : "DESC");
    rc = sqlite3_exec(db, createTableQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to create sorted table: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    // sqlite3_finalize(stmt);
    return newDf;
}

/// @brief Create a subset of a DataFrame with selected columns
/// @param df 
/// @param columns 
/// @return 
void* DataFrame_subset(void* df, const char* columns_, int szcolumns, const char* newtablename_, int sznewtablename) {
    if (df == NULL || columns_ == NULL || szcolumns <= 0 || newtablename_ == NULL || sznewtablename <= 0) {
        return 0;
    }
    char columns[szcolumns+1], newtablename[sznewtablename+1];
    memset(columns, 0, szcolumns+1);
    memset(newtablename, 0, sznewtablename+1);
    strncpy(columns, columns_, szcolumns);
    strncpy(newtablename, newtablename_, sznewtablename);

    DataFrame *df_ = (DataFrame*)df;
    char subsetQuery[1024];
    sprintf(subsetQuery, "SELECT %s FROM %s;", columns, df_->tableName);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, subsetQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare subset query: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    DataFrame *newDf = DataFrame_new(newtablename, columns);

    sqlite3_finalize(stmt);
    char createTableQuery[1024];
    sprintf(createTableQuery, "INSERT INTO %s (%s) SELECT %s FROM %s;", newDf->tableName, columns, columns, df_->tableName);
    rc = sqlite3_exec(db, createTableQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to create subset table: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    
    // fprintf(stderr, "Printing... =======\n");
    // printTableNamesAndSchema(db);
    // DataFrame_printRows(df_, 5);
    // fprintf(stderr, "Printing### =======\n");
    // printTableNamesAndSchema(db);
    // DataFrame_printRows(newDf, 5);
    return newDf;
}

/// @brief Print table names and their schema in SQLite
/// @param db Pointer to the SQLite database connection
void printTableNamesAndSchema() {
    char* query = "SELECT name, sql FROM sqlite_master WHERE type='table';";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare query: %s\n", sqlite3_errmsg(db));
        return;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* tableName = (const char*)sqlite3_column_text(stmt, 0);
        const char* schema = (const char*)sqlite3_column_text(stmt, 1);
        printf("Table Name: %s\n", tableName);
        printf("Schema: %s\n", schema);
        
        // Extract column types from schema
        char* columnTypes = strtok(schema, "(");
        columnTypes = strtok(NULL, ")");
        printf("Column Types: %s\n", columnTypes);
        
        printf("\n");
    }
    sqlite3_finalize(stmt);
}

/// @brief Print n number of rows of a DataFrame
/// @param df 
/// @param n 
void* DataFrame_printRows(void* df, int n) {
    DataFrame *df_ = (DataFrame*)df;
    char printQuery[1024];
    sprintf(printQuery, "SELECT * FROM %s LIMIT %d;", df_->tableName, n);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, printQuery, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare print query: %s\n", sqlite3_errmsg(db));
        return;
    }
    int columnCount = sqlite3_column_count(stmt);
    for (int i = 0; i < columnCount; i++) {
        printf("%s | ", sqlite3_column_name(stmt, i));
    }
    printf("\n");
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        for (int i = 0; i < columnCount; i++) {
            printf("%s | ", sqlite3_column_text(stmt, i));
        }
        printf("\n");
    }
    sqlite3_finalize(stmt);
    return df;
}


void *DataFrame_addColumn(void* df, const char* columnName_, int szcolumnName, const char* srcColumnName_, int szsrcColumnName) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 || srcColumnName_ == NULL || szsrcColumnName <= 0) {
        return NULL;
    }
    char columnName[szcolumnName+1], srcColumnName[szsrcColumnName+1];
    memset(columnName, 0, szcolumnName+1);
    memset(srcColumnName, 0, szsrcColumnName+1);
    strncpy(columnName, columnName_, szcolumnName);
    strncpy(srcColumnName, srcColumnName_, szsrcColumnName);

    DataFrame *df_ = (DataFrame*)df;
    char addColumnQuery[1024];
    sprintf(addColumnQuery, "ALTER TABLE %s ADD COLUMN %s TEXT NOT NULL DEFAULT ('');", df_->tableName, columnName);
    int rc = sqlite3_exec(db, addColumnQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_addColumn: Failed to add column: %s\n", sqlite3_errmsg(db));
        return NULL;
    }

    char updateColumnQuery[1024];
    sprintf(updateColumnQuery, "UPDATE %s SET %s = %s;", df_->tableName, columnName, srcColumnName);
    rc = sqlite3_exec(db, updateColumnQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_addColumn: Failed to update column: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df;
}


void* DataFrame_filter(void* df, 
                        const char* columnName_, int szcolumnName,
                        const char* predicate_, int szpredicate,
                        const char* value_, int szvalue,
                        const char* newTableName_, int sznewTableName
                        ) {
    if (df == NULL 
            || columnName_ == NULL || szcolumnName <= 0 
            || predicate_ == NULL || szpredicate <= 0 
            || value_ == NULL || szvalue <= 0 
            || newTableName_ == NULL || sznewTableName <= 0) {
        return NULL;
    }
    
    char columnName[szcolumnName+1], predicate[szpredicate+1], newTableName[sznewTableName+1], value[szvalue+1];
    memset(columnName, 0, szcolumnName+1);
    memset(predicate, 0, szpredicate+1);
    memset(newTableName, 0, sznewTableName+1);
    memset(value, 0, szvalue+1);
    strncpy(columnName, columnName_, szcolumnName);
    strncpy(predicate, predicate_, szpredicate);
    strncpy(newTableName, newTableName_, sznewTableName);
    strncpy(value, value_, szvalue);

    DataFrame *df_ = (DataFrame*)df;

    DataFrame *newDf = DataFrame_new_NoTabCreate(newTableName);
    char filterQuery[1024];
    sprintf(filterQuery, "CREATE TABLE %s AS SELECT * FROM %s WHERE %s %s '%s';", newDf->tableName, df_->tableName, columnName, predicate, value);
    //fprintf(stderr, "Filter Query: %s\n", filterQuery);
    int rc = sqlite3_exec(db, filterQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to execute filter query: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return newDf;
}

void* DataFrame_merge(void* df1, void* df2, const char* newTableName, int sznewTableName) {
    if (df1 == NULL || df2 == NULL || newTableName == NULL || sznewTableName <= 0) {
        return NULL;
    }
    char newTableName_[sznewTableName+1];
    memset(newTableName_, 0, sznewTableName+1);
    strncpy(newTableName_, newTableName, sznewTableName);

    DataFrame* df1_ = (DataFrame*)df1;
    DataFrame* df2_ = (DataFrame*)df2;
    
    // Create a new DataFrame for the merged table
    DataFrame* mergedDf = DataFrame_new(newTableName,NULL);
    
    // Get the column names of df1
    char columnNames1[1024];
    sprintf(columnNames1, "PRAGMA table_info(%s);", df1_->tableName);
    sqlite3_stmt* stmt1;
    int rc1 = sqlite3_prepare_v2(db, columnNames1, -1, &stmt1, NULL);
    if (rc1 != SQLITE_OK) {
        fprintf(stderr, "DataFrame_merge: Failed to prepare column names query for df1: %s\n", sqlite3_errmsg(db));
        DataFrame_free(mergedDf);
        return NULL;
    }

    // Get the column names of df2
    char columnNames2[1024];
    sprintf(columnNames2, "PRAGMA table_info(%s);", df2_->tableName);
    sqlite3_stmt* stmt2;
    int rc2 = sqlite3_prepare_v2(db, columnNames2, -1, &stmt2, NULL);
    if (rc2 != SQLITE_OK) {
        fprintf(stderr, "DataFrame_merge: Failed to prepare column names query for df2: %s\n", sqlite3_errmsg(db));
        DataFrame_free(mergedDf);
        return NULL;
    }

    char insertQuerySrc[1024], insertQueryDest[1024], temp[1024];
    memset(insertQuerySrc, 0, 1024);
    memset(insertQueryDest, 0, 1024);

    // Iterate over the column names of df2 and add any new columns to the merged table
    while (sqlite3_step(stmt1) == SQLITE_ROW) {
        const char* columnName = (const char*)sqlite3_column_text(stmt1, 1);
        int hasColumn = DataFrame_hasColumn(mergedDf, columnName, strlen(columnName));
        if (!hasColumn) {
            sprintf(temp, "%s.%s, ", df1_->tableName,columnName);
            strcat(insertQuerySrc, temp);
            sprintf(temp, "%s, ", columnName);
            strcat(insertQueryDest, temp);

            char addColumnQuery[1024];
            sprintf(addColumnQuery, "ALTER TABLE %s ADD COLUMN %s TEXT;", mergedDf->tableName, columnName);
            int rc4 = sqlite3_exec(db, addColumnQuery, NULL, NULL, NULL);
            if (rc4 != SQLITE_OK) {
                fprintf(stderr, "DataFrame_merge: Failed to add column to merged table: %s\n", sqlite3_errmsg(db));
                DataFrame_free(mergedDf);
                return NULL;
            }
        }
    }

    
    // Iterate over the column names of df2 and add any new columns to the merged table
    while (sqlite3_step(stmt2) == SQLITE_ROW) {
        const char* columnName = (const char*)sqlite3_column_text(stmt2, 1);
        int hasColumn = DataFrame_hasColumn(mergedDf, columnName, strlen(columnName));
        if (!hasColumn) {
            sprintf(temp, "%s.%s, ", df2_->tableName,columnName);
            strcat(insertQuerySrc, temp);
            sprintf(temp, "%s, ", columnName);
            strcat(insertQueryDest, temp);
            char addColumnQuery[1024];
            sprintf(addColumnQuery, "ALTER TABLE %s ADD COLUMN %s TEXT;", mergedDf->tableName, columnName);
            int rc4 = sqlite3_exec(db, addColumnQuery, NULL, NULL, NULL);
            if (rc4 != SQLITE_OK) {
                fprintf(stderr, "DataFrame_merge: Failed to add column to merged table: %s\n", sqlite3_errmsg(db));
                DataFrame_free(mergedDf);
                return NULL;
            }
        }
    }

    remove_after_last_delim(insertQuerySrc,',');
    remove_after_last_delim(insertQueryDest,',');
    
    // Merge the data from df2 into the merged table
    char mergeDataQuery[1024];
    sprintf(mergeDataQuery, "INSERT INTO %s (%s) SELECT %s FROM %s JOIN %s USING (rowid);", 
                                mergedDf->tableName, insertQueryDest,
                                insertQuerySrc,
                                df1_->tableName,
                                df2_->tableName);
    int rc5 = sqlite3_exec(db, mergeDataQuery, NULL, NULL, NULL);
    if (rc5 != SQLITE_OK) {
        fprintf(stderr, "DataFrame_merge: Failed to merge data into merged table: %s\n", sqlite3_errmsg(db));
        DataFrame_free(mergedDf);
        return NULL;
    }
    
    // Clean up
    sqlite3_finalize(stmt1);
    sqlite3_finalize(stmt2);
    
    return mergedDf;
}

void *DataFrame_groupby(void* df, const char* columnName_, int szcolumnName, const char* newTableName_, int sznewTableName, int skipColumnCheck) {
    if (df == NULL || columnName_ == NULL || szcolumnName <= 0 || newTableName_ == NULL || sznewTableName <= 0) {
        return NULL;
    }
    char newTableName[sznewTableName+1], columnName[szcolumnName+1];
    memset(newTableName, 0, sznewTableName+1);
    memset(columnName, 0, szcolumnName+1);
    strncpy(newTableName, newTableName_, sznewTableName);
    strncpy(columnName, columnName_, szcolumnName);

    DataFrame *df_ = (DataFrame*)df;

    if (!skipColumnCheck) {
        // Get the column names of the input DataFrame
        char columnNames[1024];
        sprintf(columnNames, "PRAGMA table_info(%s);", df_->tableName);
        sqlite3_stmt* stmt;
        int rc = sqlite3_prepare_v2(db, columnNames, -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            // Handle error
            fprintf(stderr, "DataFrame_groupby: Failed to prepare column names query: %s\n", sqlite3_errmsg(db));
            return NULL;
        }
        
        // Check if the specified column exists in the input DataFrame
        int columnExists = 0;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* columnNameInDF = (const char*)sqlite3_column_text(stmt, 1);
            if (strcmp(columnNameInDF, columnName) == 0) {
                columnExists = 1;
                break;
            }
        }
        sqlite3_finalize(stmt);

        if (!columnExists) {
            // Handle error - specified column does not exist in the input DataFrame
            fprintf(stderr, "DataFrame_groupby: Column does not exist in the input DataFrame\n");
            return NULL;
        }
    }
    
    // Create a new DataFrame for the grouped data
    DataFrame* groupedDf = DataFrame_new_NoTabCreate(newTableName);
    
    // Generate the GROUP BY query
    char groupByQuery[1024];
    sprintf(groupByQuery, "CREATE TABLE %s AS SELECT * FROM %s GROUP BY %s;", groupedDf->tableName, df_->tableName, columnName);
    
    // Execute the GROUP BY query
    int rc2 = sqlite3_exec(db, groupByQuery, NULL, NULL, NULL);
    if (rc2 != SQLITE_OK) {
        // Handle error
        fprintf(stderr, "DataFrame_groupby: Failed to group by column: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    
    return groupedDf;
}

void *DataFrame_AritheMeticOperation_Add(void* df, const char* srcColumnName_, int szsrcColumnName,
                                                    const char* destColumnName_, int szdestColumnName) {
    if (df == NULL || srcColumnName_ == NULL || szsrcColumnName <= 0 || destColumnName_ == NULL || szdestColumnName <= 0) {
        return NULL;
    }
    DataFrame *df_ = (DataFrame*)df;
    char srcColumnName[szsrcColumnName+1], destColumnName[szdestColumnName+1];
    memset(srcColumnName, 0, szsrcColumnName+1);
    memset(destColumnName, 0, szdestColumnName+1);
    strncpy(srcColumnName, srcColumnName_, szsrcColumnName);
    strncpy(destColumnName, destColumnName_, szdestColumnName);

    char addColumnsQuery[1024];
    sprintf(addColumnsQuery, "UPDATE %s SET %s = %s + %s;", df_->tableName, destColumnName, destColumnName,  srcColumnName);
    int rc = sqlite3_exec(db, addColumnsQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_AritheMeticOperation_Add: Failed to perform addition between columns: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df_;
}

void *DataFrame_AritheMeticOperation_ScalarAdd(void* df, const char* destColumnName_, int szdestColumnName, float scalarVal) {
    if (df == NULL || destColumnName_ == NULL || szdestColumnName <= 0) {
        return NULL;
    }
    DataFrame *df_ = (DataFrame*)df;
    char destColumnName[szdestColumnName+1];
    memset(destColumnName, 0, szdestColumnName+1);
    strncpy(destColumnName, destColumnName_, szdestColumnName);

    char addColumnsQuery[1024];
    sprintf(addColumnsQuery, "UPDATE %s SET %s = %s + %f;", df_->tableName, destColumnName,destColumnName, scalarVal);
    int rc = sqlite3_exec(db, addColumnsQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_AritheMeticOperation_ScalarAdd: Failed to perform addition between scalar and column: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df_;
}


void *DataFrame_AritheMeticOperation_Subtract(void* df, const char* srcColumnName_, int szsrcColumnName,
                                                    const char* destColumnName_, int szdestColumnName) {
    if (df == NULL || srcColumnName_ == NULL || szsrcColumnName <= 0 || destColumnName_ == NULL || szdestColumnName <= 0) {
        return NULL;
    }
    DataFrame *df_ = (DataFrame*)df;
    char srcColumnName[szsrcColumnName+1], destColumnName[szdestColumnName+1];
    memset(srcColumnName, 0, szsrcColumnName+1);
    memset(destColumnName, 0, szdestColumnName+1);
    strncpy(srcColumnName, srcColumnName_, szsrcColumnName);
    strncpy(destColumnName, destColumnName_, szdestColumnName);

    char addColumnsQuery[1024];
    sprintf(addColumnsQuery, "UPDATE %s SET %s = %s - %s;", df_->tableName, destColumnName, destColumnName,  srcColumnName);
    int rc = sqlite3_exec(db, addColumnsQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_AritheMeticOperation_Add: Failed to perform addition between columns: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df_;
}

void *DataFrame_AritheMeticOperation_ScalarSubtract(void* df, const char* destColumnName_, int szdestColumnName, float scalarVal) {
    if (df == NULL || destColumnName_ == NULL || szdestColumnName <= 0) {
        return NULL;
    }
    DataFrame *df_ = (DataFrame*)df;
    char destColumnName[szdestColumnName+1];
    memset(destColumnName, 0, szdestColumnName+1);
    strncpy(destColumnName, destColumnName_, szdestColumnName);

    char addColumnsQuery[1024];
    sprintf(addColumnsQuery, "UPDATE %s SET %s = %s - %f;", df_->tableName, destColumnName,destColumnName, scalarVal);
    int rc = sqlite3_exec(db, addColumnsQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_AritheMeticOperation_ScalarAdd: Failed to perform addition between scalar and column: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df_;
}


void *DataFrame_AritheMeticOperation_Multiply(void* df, const char* srcColumnName_, int szsrcColumnName,
                                                    const char* destColumnName_, int szdestColumnName) {
    if (df == NULL || srcColumnName_ == NULL || szsrcColumnName <= 0 || destColumnName_ == NULL || szdestColumnName <= 0) {
        return NULL;
    }
    DataFrame *df_ = (DataFrame*)df;
    char srcColumnName[szsrcColumnName+1], destColumnName[szdestColumnName+1];
    memset(srcColumnName, 0, szsrcColumnName+1);
    memset(destColumnName, 0, szdestColumnName+1);
    strncpy(srcColumnName, srcColumnName_, szsrcColumnName);
    strncpy(destColumnName, destColumnName_, szdestColumnName);

    char addColumnsQuery[1024];
    sprintf(addColumnsQuery, "UPDATE %s SET %s = %s * %s;", df_->tableName, destColumnName, destColumnName,  srcColumnName);
    int rc = sqlite3_exec(db, addColumnsQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_AritheMeticOperation_Add: Failed to perform addition between columns: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df_;
}

void *DataFrame_AritheMeticOperation_ScalarMultiply(void* df, const char* destColumnName_, int szdestColumnName, float scalarVal) {
    if (df == NULL || destColumnName_ == NULL || szdestColumnName <= 0) {
        return NULL;
    }
    DataFrame *df_ = (DataFrame*)df;
    char destColumnName[szdestColumnName+1];
    memset(destColumnName, 0, szdestColumnName+1);
    strncpy(destColumnName, destColumnName_, szdestColumnName);

    char addColumnsQuery[1024];
    sprintf(addColumnsQuery, "UPDATE %s SET %s = %s * %f;", df_->tableName, destColumnName,destColumnName, scalarVal);
    int rc = sqlite3_exec(db, addColumnsQuery, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "DataFrame_AritheMeticOperation_ScalarAdd: Failed to perform addition between scalar and column: %s\n", sqlite3_errmsg(db));
        return NULL;
    }
    return df_;
}
