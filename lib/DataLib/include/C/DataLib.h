#ifndef __DATA_LIB_H__
#define __DATA_LIB_H__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void DataFrame_deleteAllTables();
void DataFrame_free(void* df);
void* DataFrame_read_csv(const char* tableName, int sztableName, const char* csvFilePath, int szcsvFilePath);

float DataFrame_sum   (void* df, const char* columnName, int szcolumnName);
float DataFrame_avg   (void* df, const char* columnName, int szcolumnName);
int   DataFrame_count (void* df, const char* columnName, int szcolumnName);
float DataFrame_min   (void* df, const char* columnName, int szcolumnName);
float DataFrame_max   (void* df, const char* columnName, int szcolumnName);

int   DataFrame_hasColumn(void* df, const char* columnName, int szcolumnName);
void*  DataFrame_printRows(void* df, int n);
void* DataFrame_subset(void* df, const char* columns_, int szcolumns, const char* newtablename_, int sznewtablename);
void* DataFrame_sortByColumn(void* df, const char* columnName_, int szcolumnName, int ascending , const char* newtablename_, int sznewtablename);
void* DataFrame_merge(void* df1, void* df2, const char* newTableName, int sznewTableName);
void *DataFrame_groupby(void* df, const char* columnName_, int szcolumnName, const char* newTableName_, int sznewTableName, int skipColumnCheck);

void *DataFrame_addColumn(void* df, const char* columnName_, int szcolumnName, const char* srcColumnName_, int szsrcColumnName);

void *DataFrame_AritheMeticOperation_Add(void* df, const char* srcColumnName_, int szsrcColumnName, const char* destColumnName_, int szdestColumnName);
void *DataFrame_AritheMeticOperation_ScalarAdd(void* df, const char* destColumnName_, int szdestColumnName, float scalarVal);

void *DataFrame_AritheMeticOperation_Subtract(void* df, const char* srcColumnName_, int szsrcColumnName, const char* destColumnName_, int szdestColumnName);
void *DataFrame_AritheMeticOperation_ScalarSubtract(void* df, const char* destColumnName_, int szdestColumnName, float scalarVal);

void *DataFrame_AritheMeticOperation_Multiply(void* df, const char* srcColumnName_, int szsrcColumnName, const char* destColumnName_, int szdestColumnName);
void *DataFrame_AritheMeticOperation_ScalarMultiply(void* df, const char* destColumnName_, int szdestColumnName, float scalarVal);

void* DataFrame_filter(void* df, 
                        const char* columnName_, int szcolumnName,
                        const char* predicate_, int szpredicate,
                        const char* value_, int szvalue,
                        const char* newTableName_, int sznewTableName
                        );

//TODO: Add Predicate based filter
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __DATA_LIB_H__