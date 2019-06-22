package com.wolfram.databaselink;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Enumeration;
import java.util.Properties;

import org.apache.commons.dbcp.BasicDataSource;

public class JDBCConnectionManager {

    public static Connection getConnection(String url) throws SQLException {
        return DriverManager.getConnection(url);
    }

    public static Connection getConnection(String url, Properties properties, int timeout ) throws SQLException {
        int to = DriverManager.getLoginTimeout();
        if ( timeout > 0) {
        	DriverManager.setLoginTimeout(timeout);
        }
        Connection conn = DriverManager.getConnection(url, properties);
        if ( timeout > 0) {
        	DriverManager.setLoginTimeout(to);
        }
        return conn;
    }

    public static BasicDataSource getPool(String driver, String url, Properties properties) throws SQLException {        
        BasicDataSource ds = new BasicDataSource();
        ds.setDriverClassName(driver);
        ds.setUrl(url);
        Enumeration e = properties.propertyNames();
        while(e.hasMoreElements()) {
            String property = (String)e.nextElement();
            ds.addConnectionProperty(property, properties.getProperty(property));
        }            
        return ds;
    }
}