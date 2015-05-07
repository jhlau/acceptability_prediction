# A wrapper around sqlite to use it to store dictionaries

import collections

class TableDictionary(collections.MutableMapping):
    """A specialized class to implement dictionaries using strings as keys and integers as values using an sqlite table"""
    def __init__(self,connection,name):
        self.conn = connection
        self.cursor = self.conn.cursor()
        self.name = name
        if not self.table_exist():
            self.cursor.execute("CREATE TABLE " + self.name + "(ngram text, count integer)")

    def table_exist(self):
        return self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (self.name,)).fetchall() != []

    def commit_changes(self):
        self.conn.commit()

    def __len__(self):
        return self.cursor.execute("SELECT Count(*) FROM " + self.name).fetchone()[0]

    def __contains__(self,key):
        return self.cursor.execute("SELECT * FROM " + self.name + " WHERE ngram=?",(key,)).fetchall() != []

    def __iter__(self):
        for pair in self.cursor.execute("SELECT * FROM " + self.name).fetchall():
            yield pair

    def __getitem__(self,key):
        res = self.cursor.execute("SELECT count FROM " + self.name + " WHERE ngram=?",(key,)).fetchone()
        if res:
            return res[0]
        else:
            return 0 # here we behave like a defaultdict instead of raising a KeyError

    # This could probably be implemented to be faster
    def __setitem__(self,key,value):
        if key in self:
            self.cursor.execute("UPDATE " + self.name + " SET count = ? where ngram=?",(value,key))
        else:
            self.cursor.execute("INSERT INTO " + self.name + " VALUES (?,?)",(key,value))

    def __delitem__(self,key):
        self.cursor.execute("DELETE from " + self.name + " WHERE ngram=?",(key,))


class WriteOnceTableDictionary(TableDictionary):
    """This is a version of TableDictionary meant to be used if we want to write to the DB only once per key"""
    def __setitem__(self,key,value):
        self.cursor.execute("INSERT INTO " + self.name + " VALUES (?,?)",(key,value))


