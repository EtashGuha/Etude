package com.wolfram.databaselink.examples;

import java.sql.*;
import java.util.*;

import com.odellengineeringltd.glazedlists.query.*;

public class CommandQuery implements Query
{

  private Connection conn;
  private String path;

  public CommandQuery(String path)
  {
    this.path = path;
  }

  public String getName()
  {
    return "Commands";
  }

  public SortedSet doQuery() throws InterruptedException
  {

    SortedSet results = new TreeSet();
    try
    {
      conn = getConnection();
      Statement stmt = conn.createStatement();
      ResultSet rs = stmt.executeQuery("SELECT * FROM COMMANDS");
      while (rs.next())
      {
        int id = rs.getInt(1);
        String expr = rs.getString(2);
        String fullForm = rs.getString(3);
        byte[] image = rs.getBytes(4);
        Timestamp date = rs.getTimestamp(5);
        results.add(new Command(id, expr, fullForm, image, date));
      }
      return results;
    }
    catch(ClassNotFoundException e)
    {
      e.printStackTrace();
      return results;
    }
    catch(SQLException e)
    {
      e.printStackTrace();
      return results;
    }

  }

    private Connection getConnection() throws ClassNotFoundException, SQLException
    {
      if(conn != null)
        return conn;

      Class.forName("org.hsqldb.jdbcDriver");
      return DriverManager.getConnection("jdbc:hsqldb:" + path, "sa", "");
    }

    public boolean matchesObject(Comparable object) {
        return (object instanceof Command);
    }
}