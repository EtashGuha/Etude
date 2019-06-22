package com.wolfram.databaselink;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.Date;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Timestamp;
import java.sql.Types;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.TimeZone;
import java.util.regex.Pattern;

import com.wolfram.jlink.Expr;
import com.wolfram.jlink.ExprFormatException;

public class SQLStatementProcessor {

	private static final Expr SYM_SQLBINARY = new Expr(Expr.SYMBOL, "SQLBinary");
	private static final Expr SYM_DatabaseLinkSQLBINARY = new Expr(Expr.SYMBOL, "DatabaseLink`SQLBinary");
	private static final Expr SYM_SQLDATETIME = new Expr(Expr.SYMBOL, "SQLDateTime");
	private static final Expr SYM_DatabaseLinkSQLDATETIME = new Expr(Expr.SYMBOL, "DatabaseLink`SQLDateTime");
	private static final Expr SYM_DATEOBJECT = new Expr(Expr.SYMBOL, "DateObject");
	private static final Expr SYM_TIMEOBJECT = new Expr(Expr.SYMBOL, "TimeObject");
	private static final Expr SYM_RULE = new Expr(Expr.SYMBOL, "Rule");
	private static final Expr SYM_SQLEXPR = new Expr(Expr.SYMBOL, "SQLExpr");
	private static final Expr SYM_DatabaseLinkSQLExpr = new Expr(Expr.SYMBOL, "DatabaseLink`SQLExpr");
	private static final Expr SYM_NULL = new Expr(Expr.SYMBOL, "Null");

    public static Object[] processSQLStatement(
        Connection connection,
        String sql,
        Expr params,
        boolean getAsStrings,
        boolean showColumnHeadings,
        boolean returnResultSet,
        boolean returnGeneratedKeys,
        boolean clearParams,
        int maxrows,
        int timeout,
        int resultSetType,
        int resultSetConcurrency, 
        int escapeProcessing,
        int fetchDirection, 
        int fetchSize,
        int maxFieldSize,
        int batchSize) throws Exception {
        PreparedStatement ps = null;
        
        int defaultYear = 0;
        int defaultMonth = 1;
        int defaultDate = 1 ;
        int defaultHour = 0;
        int defaultMinute = 0;
        double defaultSeconds = 0.0;

        boolean noParams = false;
        
        if (returnGeneratedKeys) {
            ps = connection.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS);
        } else {
            ps = connection.prepareStatement(sql, resultSetType, resultSetConcurrency);
        }
    
        boolean k = false;
        //int [] intArray = null;
        ArrayList<Integer> res = new ArrayList<Integer>();
        ArrayList<Object> keys = new ArrayList<Object>();
        boolean success = true;     // for batch adds
    
        if (maxrows > 0)
          ps.setMaxRows(maxrows);
    
        if (timeout > 0)
          ps.setQueryTimeout(timeout);
    
        if (fetchDirection > 0)
            ps.setFetchDirection(fetchDirection);
    
        if (fetchSize != 0)
            ps.setFetchSize(fetchSize);
    
        if (maxFieldSize > 0)
            ps.setMaxFieldSize(maxFieldSize);
        
        if (escapeProcessing >= 0)
            // Setting this to false appears to be pointless for prepared statements
            ps.setEscapeProcessing((escapeProcessing == 1) ? true:false);
    
        // TODO Mem use implications of this loop
		int len = params.length();
		for (int h = 1; h <= len; h++) {
			Expr list = params.part(h);
			if (list.length() != 0) {
				for (int i = 1; i <= list.length(); i++) {
					Expr e = list.part(i);
					if (e.realQ())
						ps.setDouble(i, e.asDouble());
					else if (e.integerQ())
						ps.setLong(i, e.asLong());
					else if (e.stringQ())
						ps.setString(i, e.asString());
					else if (e.equals(Expr.SYM_TRUE) || e.equals(Expr.SYM_FALSE))
						ps.setBoolean(i, e.trueQ());
					else if (e.equals(SYM_NULL))
						ps.setNull(i, Types.NULL);

					else if (isSQLBinary(e.head())) {
						if (e.part(1).vectorQ(Expr.INTEGER)) {
							int[] a = (int[]) e.part(1).asArray(Expr.INTEGER, 1);
							byte[] bytes = new byte[a.length];
							for (int j = 0; j < a.length; j++) {
								if (a[j] > 127)
									bytes[j] = (byte) (a[j] - 256);
								else
									bytes[j] = (byte) a[j];
							}
							ps.setBytes(i, bytes);
						} else {
							byte[] bytes = new byte[e.length()];
							for (int j = 1; j <= e.length(); j++) {
								Expr a = e.part(j);
								if (a.integerQ()) {
									int b = a.asInt();
									if (b > 127)
										bytes[j - 1] = (byte) (b - 256);
									else
										bytes[j - 1] = (byte) b;
								} else {
									throw new Exception("SQLBinary may only contain integers from 0 to 255.");
								}
							}
							ps.setBytes(i, bytes);
						}
					} else if (e.head().equals(SYM_RULE)) {

					} else if (e.head().equals(SYM_DATEOBJECT)) {
						/*
						 * Format should be DateObject[list, granularity,
						 * calendarType, timezone] where list is a list of 1 to
						 * 6 elements.
						 */
						if (e.length() != 4) {
							throw new Exception("Illegal value for DateObject: " + e.toString());
						}

						Calendar cal = Calendar.getInstance();
						Expr edate = e.part(1);
						Expr calendarType = e.part(3);
						String timezone = normalizeTimeZone(e.part(4));
						int nanval = 0;

						if (!calendarType.asString().equals("Gregorian")) {
							throw new Exception("Only Gregorian calender is supported");
						}

						if (!edate.listQ() || edate.length() < 1) {
							throw new Exception("Illegal value for DateObject: " + edate.toString());
						}

						TimeZone timeZone = TimeZone.getTimeZone(timezone);
						cal.setTimeZone(timeZone);

						int valYear = getIntegerDateValue(edate, 1, "year", defaultYear);
						cal.set(Calendar.YEAR, valYear);

						int valMonth = getIntegerDateValue(edate, 2, "month", defaultMonth);
						cal.set(Calendar.MONTH, valMonth - 1);

						int valDate = getIntegerDateValue(edate, 3, "date", defaultDate);
						cal.set(Calendar.DATE, valDate);

						int valHour = getIntegerDateValue(edate, 4, "hour", defaultHour);
						cal.set(Calendar.HOUR_OF_DAY, valHour);

						int valMinute = getIntegerDateValue(edate, 5, "minute", defaultMinute);
						cal.set(Calendar.MINUTE, valMinute);

						double valSeconds = getRealDateValue(edate, 6, "second", defaultSeconds);
						int secval = new Double(valSeconds).intValue();
						nanval = new Double((valSeconds - secval) * 1000000000).intValue();
						cal.set(Calendar.SECOND, secval);

						long time = cal.getTime().getTime();
						Timestamp ts = new Timestamp(cal.getTime().getTime());
						ts.setNanos(nanval);
						ps.setTimestamp(i, ts);

					} else if (e.head().equals(SYM_TIMEOBJECT)) {

						if (e.length() < 3) {
							throw new Exception("Illegal value for TimeObject: " + e.toString());
						}

						Calendar cal = Calendar.getInstance();
						Expr etime = e.part(1);
						int nanval = 0;

						int valHour = getIntegerDateValue(etime, 1, "hour", defaultHour);
						cal.set(Calendar.HOUR_OF_DAY, valHour);

						int valMinute = getIntegerDateValue(etime, 2, "minute", defaultMinute);
						cal.set(Calendar.MINUTE, valMinute);

						double valSeconds = getRealDateValue(etime, 3, "second", defaultSeconds);
						int secval = new Double(valSeconds).intValue();
						nanval = new Double((valSeconds - secval) * 1000000000).intValue();
						cal.set(Calendar.SECOND, secval);

						Expr timeZoneExpr = e.part(3);
						if (timeZoneExpr.toString().equals("None")) {
							cal.setTimeZone(TimeZone.getDefault());
						} else {
							cal.setTimeZone(TimeZone.getTimeZone(normalizeTimeZone(timeZoneExpr)));
						}

						Timestamp ts = new Timestamp(cal.getTime().getTime());
						ts.setNanos(nanval);
						ps.setTimestamp(i, ts);

					} else if (isSQLDateTime(e.head())) {

						if (!e.part(1).listQ() || e.part(1).length() < 1) {
							throw new Exception("Illegal values for SQLDateTime: " + e.part(1).toString());
						}

						e = e.part(1);
						Calendar cal = Calendar.getInstance();
						int nanval = 0;

						int valYear = getIntegerDateValue(e, 1, "year", defaultYear);
						cal.set(Calendar.YEAR, valYear);

						int valMonth = getIntegerDateValue(e, 2, "month", defaultMonth);
						cal.set(Calendar.MONTH, valMonth - 1);

						int valDate = getIntegerDateValue(e, 3, "date", defaultDate);
						cal.set(Calendar.DATE, valDate);

						int valHour = getIntegerDateValue(e, 4, "hour", defaultHour);
						cal.set(Calendar.HOUR_OF_DAY, valHour);

						int valMinute = getIntegerDateValue(e, 5, "minute", defaultMinute);
						cal.set(Calendar.MINUTE, valMinute);

						double valSeconds = getRealDateValue(e, 6, "second", defaultSeconds);
						int secval = new Double(valSeconds).intValue();
						nanval = new Double((valSeconds - secval) * 1000000000).intValue();
						cal.set(Calendar.SECOND, secval);

						Timestamp ts = new Timestamp(cal.getTime().getTime());
						ts.setNanos(nanval);
						ps.setTimestamp(i, ts);
						// }
					} else if (isSQLExpr(e.head())) {
						ps.setString(i, e.part(1).asString());
					} else {
						throw new Exception("Illegal value: " + e.toString());
					}
				} // loop over row contents (columns)
			} else {
				noParams = true;
			}
			if (batchSize > 0 && params.length() > 1) {
				try {
					ps.addBatch();
				} catch (Exception e) {
					success = false;
					batchSize = 0;
					k = ps.execute(sql);
					if (!k) {
						res.add(ps.getUpdateCount());
					}
				} finally {
					if (success && ((h % batchSize == 0) || (h == params.length()))) {
						// executeBatch() returns an array of update counts
						int[] updateCounts = ps.executeBatch();
						if (returnGeneratedKeys) {
							ResultSet keyRs = ps.getGeneratedKeys();
							Object[] latestKeys = getAllResultData(keyRs, getAsStrings, showColumnHeadings);
							for (Object key : latestKeys) {
								keys.add(key);
							}
						} else {
							for (int i : updateCounts) {
								res.add(i);
							}
						}
					}
				}
			} else {
				k = ps.execute();
				if (!k) {
					if (returnGeneratedKeys) {
						ResultSet keyRs = ps.getGeneratedKeys();
						Object[] latestKeys = getAllResultData(keyRs, getAsStrings, showColumnHeadings);
						for (Object key : latestKeys) {
							keys.add(key);
						}
					} else {
						res.add(ps.getUpdateCount());
					}
				}
			}
			if (clearParams == true && noParams == false) {
				ps.clearParameters();
			}
		} // loop over rows (params)

		ResultSet rs = null;
		/* Generated Keys (this will ignore returnResultSet) */
		if (returnGeneratedKeys) {
			ps.close();
			return keys.toArray(new Object[keys.size()]);
		} /* select statements */ else if (k) {
			rs = ps.getResultSet();
			if (returnResultSet) {
				return new ResultSet[] { rs };
			}
			Object[] results = getAllResultData(rs, getAsStrings, showColumnHeadings);
			ps.close();
			return results;
		} /* batch statements */ else if (!res.isEmpty()) {
			Integer[] integerArray = new Integer[res.size()];
			res.toArray(integerArray);
			ps.close();
			return integerArray;
		}
		/* insert, update, remove statements */
		int updateCount = ps.getUpdateCount();
		ps.close();
		return new Integer[] { new Integer(updateCount) };
	}

    public static String normalizeTimeZone(Expr timezone) throws ExprFormatException {
    	final String doublePattern = "^([-+])?([0-9]*)\\.([0-9]*)?";
    	String timezoneStr = timezone.toString();
    	boolean match = Pattern.matches(doublePattern, timezoneStr);
    	if (match == true) {
    		String[] timezoneArr = timezoneStr.split("\\.");
    		if(timezoneArr[1].equals("0")){
    			timezoneArr[1] = "00";
    		}
    		return "GMT"+timezoneArr[0]+":"+timezoneArr[1];
    	} else {
    		return timezoneStr.replace("\"", "");
    	}
    }
    
    public static double getRealDateValue( Expr edate, int index, String type, double defaultValue) throws Exception {
    	  
    	  if (edate.length() < index) {
      		  return defaultValue;
      	  }
  	
    	  if ( edate.part(index).realQ()) {
    		  return edate.part(index).asDouble();
    	  }
    	  return (double)getIntegerDateValue( edate, index, type,0);
    }
    
    public static int getIntegerDateValue( Expr edate, int index, String type, int defaultValue) throws Exception {
   
  	  if (edate.length() < index) {
  		  return defaultValue;
  	  }
  	  
  	  if (!edate.part(index).integerQ()) {
  	  throw new Exception( "Illegal value for " + type + " in DateObject: " + edate.part(index).toString());
  	  }
  	  return edate.part(index).asInt();
    }

	public static boolean isSQLBinary(Expr e) {
		return e.equals(SYM_SQLBINARY) || e.equals(SYM_DatabaseLinkSQLBINARY);
	}

	public static boolean isSQLDateTime(Expr e) {
		return e.equals(SYM_SQLDATETIME) || e.equals(SYM_DatabaseLinkSQLDATETIME);
	}

	public static boolean isSQLExpr(Expr e) {
		return e.equals(SYM_SQLEXPR) || e.equals(SYM_DatabaseLinkSQLExpr);
	}

  /* Introduced to support the streaming result set settings for MySQL
   * (prepared statements don't work for this). Overkill but might have future applications.
   * 26 Sept 2013 | dillont
   */
  public static Object[] processUnpreparedSQLStatement(
          Connection connection,
          String sql,
          boolean getAsStrings,
          boolean showColumnHeadings,
          boolean returnResultSet,
          boolean returnGeneratedKeys,
          int maxrows,
          int timeout,
          int resultSetType,
          int resultSetConcurrency, 
          int escapeProcessing,
          int fetchDirection, 
          int fetchSize,
          int maxFieldSize) throws Exception {
      Statement s = null;
      s = connection.createStatement(resultSetType, resultSetConcurrency);

      boolean k = false;
      //ArrayList<Integer> res = new ArrayList<Integer>();

      if (maxrows > 0)
          s.setMaxRows(maxrows);

      if (timeout > 0)
          s.setQueryTimeout(timeout);

      if (fetchDirection > 0)
          s.setFetchDirection(fetchDirection);

      if (fetchSize != 0)
          s.setFetchSize(fetchSize);

      if (maxFieldSize > 0)
          s.setMaxFieldSize(maxFieldSize);
          
      if (escapeProcessing >= 0)
          s.setEscapeProcessing((escapeProcessing == 1) ? true:false);

      // Accommodate SQLite supported interface
      //k = s.execute(sql, returnGeneratedKeys ? Statement.RETURN_GENERATED_KEYS:Statement.NO_GENERATED_KEYS);
      if (returnGeneratedKeys) {
          k = s.execute(sql, Statement.RETURN_GENERATED_KEYS);
      } else {
          k = s.execute(sql);
      }

      ResultSet rs = null;
      /* Generated Keys */
      if (returnGeneratedKeys) {
          rs = s.getGeneratedKeys();
          if (returnResultSet) {
              return new ResultSet[] { rs };
          }
          Object[] results = getAllResultData(rs, getAsStrings, showColumnHeadings);
          s.close();
          return results;
      }
      /* select statements */
      else if (k)
      {
          rs =  s.getResultSet();
          if (returnResultSet)
              return new ResultSet[] { rs };
          Object[] results = getAllResultData(rs, getAsStrings, showColumnHeadings);
          s.close();
          return results;
      }
      /* insert, update, remove statements */
      int updateCount = s.getUpdateCount();
      s.close();
      return new Integer[] { new Integer(updateCount) };
  }

  
  public static Object[] getHeadings(ResultSet rs, boolean tables) throws Exception
  {
      ResultSetMetaData meta = rs.getMetaData();
      Object[] headings = new Object[meta.getColumnCount()];
      for (int i = 0; i < meta.getColumnCount(); i++)
      {
        if (tables)
        {
            String[] col = new String[2];
            col[0] = meta.getTableName(i+1);
            // getColumnName will ignore aliases in MySQL | dillont
            col[1] = meta.getColumnLabel(i+1);
            headings[i] = col;
        }
        else
          // getColumnName will ignore aliases in MySQL | dillont
          headings[i] = meta.getColumnLabel(i+1);
      }      
      return headings;
  }
  
  public static Object[] getLimitedResultData(
          int limit, 
          ResultSet rs, 
          boolean getAsStrings) throws Exception
  {
      
    ArrayList<Object> data = new ArrayList<Object>();
    int[] columnTypes = getColumnTypes(rs);
    
    boolean valid = false;    
    if (limit == 0)
    {
      Object[] row = getRow(rs, columnTypes, getAsStrings);
      data.add(row);        
    }
    if (limit > 0)
    {
      Object[] row;
      for (int j = 0; j < limit; j++)
      {
        valid = rs.next();
        if (!valid)
          break;
        row = getRow(rs, columnTypes, getAsStrings);
        data.add(row);
      }
    }
    if (limit < 0)
    {    
      Object[] row;
      for (int k = 0; k > limit; k--)
      {
        valid = rs.previous();
        if (!valid)
          break;
        row = getRow(rs, columnTypes, getAsStrings);
        data.add(row);            
      }        
    }        
    if (data.size() == 0 && !valid)
      return null;
      
    return data.toArray(new Object[data.size()]);
  }

  public static Object[] getAllResultData(
          ResultSet rs, 
          boolean getAsStrings, 
          boolean showColumnHeadings) throws Exception
  {
    boolean valid = rs.next();
    
    ArrayList<Object> data = new ArrayList<Object>();
    if (showColumnHeadings)
        data.add(getHeadings(rs, false));
    int[] columnTypes = getColumnTypes(rs);
    
    while (valid) {
      Object[] row = getRow(rs, columnTypes, getAsStrings);
      data.add(row);
      valid = rs.next();
    }
    
    return data.toArray(new Object[data.size()]);
  }
  
    private static Object[] getRow(ResultSet rs, int[] columnTypes, boolean getAsStrings) throws Exception {
        int cc = columnTypes.length;
        Object[] row = new Object[cc];
        if (getAsStrings) {
            for (int j = 0; j < cc; j++) {
                row[j] = rs.getString(j + 1);
            }
        } else {
            for (int j = 0; j < cc; j++) {
                int ct = columnTypes[j];
                if (ct == Types.INTEGER || ct == Types.BIT || ct == Types.BOOLEAN || ct == Types.FLOAT || ct == Types.DOUBLE || 
                        ct == Types.BIGINT || ct == Types.REAL || ct == Types.SMALLINT || ct == Types.TINYINT || 
                        ct == Types.NUMERIC || ct == Types.DECIMAL) {
                    row[j] = rs.getObject(j + 1);
                } else if (ct == Types.BINARY || ct == Types.VARBINARY || ct == Types.LONGVARBINARY || ct == Types.BLOB) {
                    byte[] bytes = rs.getBytes(j + 1);
                    if (bytes != null) {
                        int[] a = new int[bytes.length];
                        for (int k = 0; k < bytes.length; k++) {
                            if (bytes[k] < 0) {
                                a[k] = bytes[k] + 256;
                            } else {
                                a[k] = bytes[k];
                            }
                        }
                        row[j] = new Expr(new Expr(Expr.SYMBOL, "SQLBinary"), new Expr[] {new Expr(a)});
                    } else {
                        row[j] = SYM_NULL;
                    }
                } else if (ct == Types.DATE) {
                    Date d = rs.getDate(j + 1);
                    if (d != null) {
                        Calendar cal = Calendar.getInstance();
                        cal.setTime(new Date(d.getTime()));
                        row[j] = new Expr(
                            new Expr(Expr.SYMBOL, "SQLDateTime"),
                            new Expr[] {
                                new Expr (
                                    new Expr(Expr.SYMBOL, "List"),
                                    new Expr[] {
                                        new Expr(cal.get(Calendar.YEAR) * (
                                            cal.get(Calendar.ERA) == java.util.GregorianCalendar.BC ?
                                                -1:1
                                        )) ,
                                        new Expr(cal.get(Calendar.MONTH) + 1),
                                        new Expr(cal.get(Calendar.DATE))
                                    }
                                )
                            }
                        );
                    } else {
                        row[j] = SYM_NULL;
                    }
                } 
                else if (ct == Types.TIME) {
                   Timestamp t = rs.getTimestamp(j + 1);
                    if (t != null) {
                        Calendar cal = Calendar.getInstance();
                        cal.setTime(new Date(t.getTime()));
                        row[j] = new Expr(
                            new Expr(Expr.SYMBOL, "SQLDateTime"),
                            new Expr[] {
                                new Expr(
                                    new Expr(Expr.SYMBOL, "List"),
                                    new Expr[] {
                                        new Expr(cal.get(Calendar.HOUR_OF_DAY)),
                                        new Expr(cal.get(Calendar.MINUTE)),
                                        new Expr(cal.get(Calendar.SECOND) + (new Integer(t.getNanos()).doubleValue() / 1000000000))
                                    }
                                )
                            }
                        );
                    } else {
                        row[j] = SYM_NULL;
                    }
                } 
                else if (ct == Types.TIMESTAMP) {
                    Timestamp ts = rs.getTimestamp(j + 1);
                    if (ts != null) {
                        Calendar cal = Calendar.getInstance();
                        cal.setTime(new Date(ts.getTime()));
                        row[j] = new Expr(
                            new Expr(Expr.SYMBOL, "SQLDateTime"),
                            new Expr[] {
                                new Expr(
                                    new Expr(Expr.SYMBOL, "List"),
                                    new Expr[] {
                                        new Expr(cal.get(Calendar.YEAR) * (
                                            cal.get(Calendar.ERA) == java.util.GregorianCalendar.BC ?
                                                -1:1
                                        )),
                                        new Expr(cal.get(Calendar.MONTH) + 1),
                                        new Expr(cal.get(Calendar.DATE)),
                                        new Expr(cal.get(Calendar.HOUR_OF_DAY)),
                                        new Expr(cal.get(Calendar.MINUTE)),
                                        new Expr(cal.get(Calendar.SECOND) + (new Integer(ts.getNanos()).doubleValue() / 1000000000))
                                    }
                                )
                            }
                        );
                    } else {
                        row[j] = SYM_NULL;
                    }
                } 
                else {
                    String val = rs.getString(j + 1);
                    if (val != null && val.startsWith("SQLExpr[")) {
                        row[j] = new Expr(
                            new Expr(Expr.SYMBOL, "ToExpression"),
                            new Expr[] { new Expr(val) }
                        );
                    } else {
                        row[j] = val;
                    }
                }
            } // cc (columns)
        } // if (getAsStrings)
        return row;
    }
  
  private static int[] getColumnTypes(ResultSet rs) throws Exception {
      ResultSetMetaData meta = rs.getMetaData();
      int cc = meta.getColumnCount();
      int[] columnTypes = new int[cc];
      for (int j = 0; j < cc; j++) {
          columnTypes[j] = meta.getColumnType(j + 1);
      }
      return columnTypes;
  }
  
  public static Object[] getConnectionMetaData(Connection conn) throws Exception
  {
	  Object[] data = new Object[2];

	  int mdiCount = MetaDataItem.values().length;
	  Object[] mdiNames = new Object[mdiCount];
	  Object[] mdiValues = new Object[mdiCount];
	  
	  DatabaseMetaData metaData = conn.getMetaData();
	  
	  int itemCount = 0;
	  for (MetaDataItem mdi : MetaDataItem.values())
	  {
		  mdiNames[itemCount] = mdi.name();
		  mdiValues[itemCount] = mdi.getValue(metaData);
		  itemCount++;
	  }
	  
	  data[0] = mdiNames;
	  data[1] = mdiValues;
	  
	  return data;
  }

  public static Object[] getConnectionMetaData(Connection conn, String[] mdiListRequested) throws Exception
  {
	  Object[] data = new Object[2];
  	  
	  int mdiListCount = mdiListRequested.length;
	  Object[] mdiNames = new Object[mdiListCount];
	  Object[] mdiValues = new Object[mdiListCount];
	  
	  DatabaseMetaData metaData = conn.getMetaData();
	  
	  for (int itemCount = 0; itemCount < mdiListCount; itemCount++)
	  {
		  MetaDataItem mdi = MetaDataItem.valueOf(mdiListRequested[itemCount]);

		  mdiNames[itemCount] = mdi.name();
		  mdiValues[itemCount] = mdi.getValue(metaData);
		  
	  }
	  
	  data[0] = mdiNames;
	  data[1] = mdiValues;
	  
	  return data;	  
  }

  public static Object getConnectionMetaData(Connection conn, String mdiRequested) throws Exception
  {
	  Object data = new Object();
	  
	  DatabaseMetaData metaData = conn.getMetaData();
	  
	  MetaDataItem mdi = MetaDataItem.valueOf(mdiRequested);
	  
	  data = mdi.getValue(metaData);
	  
	  return data;
  }
  
}