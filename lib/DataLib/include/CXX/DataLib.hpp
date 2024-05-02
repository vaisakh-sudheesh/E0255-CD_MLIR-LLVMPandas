#ifndef DATALIB_H
#define DATALIB_H

#include <vector>
#include <string>
#include "sqlite3.h"

class DataFrame {
public:
    DataFrame(std::string name = "data");
    
    ~DataFrame();

    bool read_csv(const std::string& csvFilePath);

    void addColumn(const std::string& , const std::string&);
    bool hasColumn(const std::string&);
    void dumpSchemas();

    int  getNumColumns ();

    int getColumnSize(const std::string&);
    std::string getColumnValue(const std::string& , int );
    std::string getColumnValue(const std::string& );
    void addData(const std::string& , const std::string& );

    DataFrame subset(const std::vector<std::string>& columns);
    std::vector<std::string> applyRowWiseOperation(const std::string& , const std::function<std::string(const std::string&)>& );
    //DataFrame groupBy(const std::vector<std::string>& groupByColumns);
    DataFrame filter(const std::string& , const std::string& , const std::string& );
    std::string sum(const std::string& );
    std::string mean(const std::string& );
    std::string count(const std::string& );
    DataFrame join(const DataFrame& , const std::string& , const std::string& ) ;
    DataFrame sortByColumn(const std::string& );

    std::string getTableName();
    void setTableName(const std::string&);

private:
    sqlite3* db;
    std::string tableName;

    // Add other private members as needed
};

#endif // DATALIB_H