# DatabaseLink

<h3>Revision History</h3>
<table border=1 cellpadding=5 cellspacing=0 summary="">
<tr>
<td>3.1.0</td><td>10.4</td><td>2/16</td><td>DateObject support, driver updates, point bugfixes and documentation updates.</td>
</tr>
<tr>
<td>3.0.0</td><td>10.0</td><td>1/14</td><td>Numerous bugfixes and gui revisions, added support for H2, Derby, SQLite, and Firebird.</td>
</tr>
<tr>
<td>2.0.0</td><td></td><td>3/06</td><td>Added connection pool feature, added additional configuration options, added additional information functions, added support for generated keys, upgraded drivers, and fixed some bugs.</td>
</tr>
<tr>
<td>1.1.0</td><td></td><td>7/05</td><td>Enhanced the SQLConnection Wizard, upgraded drivers, and fixed some bugs.</td>
</tr>
<tr>
<td>1.0.2</td><td></td><td>6/05</td><td>Minor bug fixes.</td>
</tr>
<tr>
<td>1.0.1</td><td></td><td>1/05</td><td>Minor bug fixes and SQLResultSet functionality.</td>
</tr>
<tr>
<td>1.0.0</td><td></td><td>11/04</td><td>First release.</td>
</tr>
</table>
<P>
<br>

<h3>Changes in version 3.1.0 (10.4) </h3>
<p> Git migration. </p>
<p> Added tutorial section on manual driver installation. </p>
<p> Initial implementation of InstallJDBCDriver[] (futurized). </p>
<p> Implemented feature 292825, Support for Date- and TimeObject in SQLExecute and SQLInsert. </p>
<p> Fix for bug 306014, "GeneratedKeys" returning single values only. </p>
<p> Fix for bug 306091, support for BLOB/VARBINARY column type. </p> 
<p> Overhaul of testing framework. </p>
<p> Audit for bad ReplaceAll usage. </p>
<p> Documentation polish. </p> 
<p> Support for latest version of embedded Derby. </p>
<p> Upgraded all drivers; added Drizzle, MariaDB. </p>

<h3>Changes in version 3.0.0 </h3>
<p> Upgraded all drivers.</p>
<p> Added drivers/support for H2, Derby, Firebird, and SQLite.</p>
<p> Added "Mode" -> "MySQLStreaming" for streaming result sets in MySQL.</p>
<p> SQLArgument now supports lists.</p>
<p> Configurations set to Password -> $Prompt no longer leak password value.</p> 
<p> SQLConnectionOpenQ and SQLConnectionUsableQ added for ascertaining connection status.</p>
<p> Warning added for attempted non-paramterized multirow insert with Access and Excel.</p>
<p> Fixed setLong bug affecting large integer inserts in Access and Excel. </p>
<p> HSQL databases now automatically shut down when the last connection is closed, releasing lock files. </p>
<p> Special (URL) characters now allowed in server names. </p>
<p> Option lists now supported for all DatabaseLink functions. </p>
<p> Added Index option to SQLCreateTable. </p>
<p> Added Default and PrimaryKey options to SQLColumn. </p>
<p> Added support for column aliases. </p>
<p> Fixed cases where SQLDateTime was adding the current time to its argument. </p>
<p> Fixed units bug in Timeout. </p>
<p> Added BatchSize, JavaBatching options to permit finer control over memory use and performance on large inserts. </p>
<p> Revisions to Data Source Wizard and DatabaseExplorer guis. </p>
<p> Data Source Wizard no longer corrupts IP hostnames. </p>
<p> Fixed bug in assignments to $SQLTimeout. </p>
<p> Fixed bug in FetchSize option of SQLResultSetOpen. </p>


<h3>Changes in version 2.0.0 </h3>
<p>Upgraded to MySQL 3.1.12 driver.</p>
<p>Upgraded to HSQL 1.8.0.7.</p>
<p>Upgraded to jTDS 1.2 driver.</p>
<p>Fixed a bug where large integers passed as parameters to SQLExecute, SQLInsert, etc.  were not converted 
into the proper Java objects.</p>
<p>Transactions are now cleaned up after a connection is closed, so a new transaction can be created.</p>
<p>Fixed a bug that prevented $SQLTimeout from working properly.</p>
<p>Added UseConnectionPool option to OpenSQLConnection that determines whether connection pool technology 
should be used to manage connections.  This allows users to reuse connections to enhance performance.</p>
<p>Added $SQLUseConnectionPool to determine whether a connection pool is used by default.</p>
<p>Added Properties option to OpenSQLConnection that allows the user to specify connection properties outside
the URL.  Added an interface to this option in the SQLConnection Wizard.</p>
<p>Added ReadOnly, TransactionIsolationLevel, and Catalog as options of OpenSQLConnection.  Also added 
SetSQLConnectionOptions and SetSQLConnectionPoolOptions that take these same options to set these dynamically.  
Added an interface to these options in the SQLConnection Wizard.</p>
<p>Added MaxFieldSize, FetchDirection, FetchSize, and EscapeProcessing as options to SQLSelect and SQLExecute.  
Also added SetSQLResultSetOptions that take FetchSize and FetchDirection as options to set dynamically.</p>
<p>Fixed a bug that left Java statements open. These need to be closed to free resources
for executing additional statements.  This bug only occurred for insert, update, and remove statements.</p>
<p>Updated the password prompt to include the username so it looks more familiar and may be changed via a user 
interface.</p>
<p>Updated the SQLConnection GUI to use SQLConnection wizard for editing existing connections.</p>
<p>Updated the JDBCDriver GUI to use JDBCDriver wizard for editing existing driver configurations.</p>
<p>Added ReturnGeneratedKeys option to SQLInsert and SQLExecute for returning keys that are generated for inserts.</p>
<p>Added ColumnSymbols option to SQLSelect and SQLExecute for assigning vectors containing column values to symbols.</p>
<p>Added SQLConnectionInformation function that returns information about a connection.</p>
<p>Added SQLColumnPrivileges and SQLTablePrivileges functions that returns information the access rights of columns
and tables respectively.</p>
<p>Added SQLTableExportedKeys and SQLTableImportedKeys functions that returns information about foreign keys.</p>
<p>Added SQLTablePrimaryKeys function that returns information about primary keys.</p>
<p>Added SQLTableIndexInformation function that returns information about indices.</p>
<p>Updated AddDatabaseResources and DatabaseResourcesPath to use ResourceLocator Package.  This affects a couple of 
functions such as DataSources, JDBCDrivers, and SQLQueries.  This in turn fixed a bug with AddDatabaseResources which did
not add a new path properly.</p>
<p>Added functionality that makes sure the SQLConnections[] and SQLQueries[] are not reset when the package is reloaded.</p>
<p>Fixed a bug with generating SQL conditions that occurred if logical operators such as Or, And, Equal, etc were too 
long.  For instance, if an Or expression had 40 parameters, it would fail.</p>
<p>Fixed a bug in SQLConnection wizard that prevented saving connection configurations, if the DatabaseResources 
directory did not already exist.</p>
<p>Added NewDataSource function for storing a new SQLConnection configuration.  This has defaults that make it very easy
to create a new HSQL database configuration.</p>
<p>Added support for launching a SQLServer.  Added SQLServerLaunch for launching a server, SQLServerShutdown for shutting 
it down, and SQLServerInformation for getting information about the server.  Each server launched is an HSQL server.  And each
server may be connected to over the network.  It is a nice way for Mathematica users to share their databases.</p>
<p>Added more error checking to all functions.  More messages and failures may result, but better output should be provided.</p>
<p>Removed SQLTableNames[conn, tablename].  There is no need for a function which returns its parameter.</p>
<p>Added SQLSchemaInformation for retrieving information about a schema.</p>
<p>Fixed SQLSchemaNames to only return the names of schema.  It was returning more than just the names.</p>
<p>Added support for multiple tables in SQLUpdate.</p>

<h3>Changes in version 1.1.0 </h3>
<p>Upgraded to MySQL 3.1.10 driver.</p>
<p>Upgraded to HSQL 1.8.0.0.</p>
<p>Upgraded to jTDS 1.1 driver.</p>
<p>Fixed a bug with SQLExpr that resulted in an expr returning as a string rather than an expression.</p>
<p>Enhanced the SQLConnection wizard to be easier to use. There are many more preconfigured connection types, 
each preconfigured connection type has its own configuration page, added a test button to test the connection 
before it is saved, added a connection properties page, and simplified choices for storing the SQL connection.</p>

<h3>Changes in version 1.0.2 </h3>
<p>Changed the default ResultSet type to be ForwardOnly for non-ResultSet queries.  This improves the speed of 
queries quite a bit.</p>
<p>Changed the default Timeout and MaxRows values.  The old value caused problems with Oracle 10g connections.</p>
<p>Fixed a bug with SQLSelect that caused the wrong result when using x < SQLColumn < y and x > SQLColumn > y.</p>
<p>Added functionality for support for SQLSelect when using x <= SQLColumn <= y and x => SQLColumn => y.</p>
<p>Added code to limit the amount of time metadata is retrieved for a result set.  This increases the speed 
of queries slightly.</p>
<p>Added code that enables the use of unsigned integers with MySQL.</p>
<p>Fixed a small bug that caused an error when receiving no results for a query using the ODBC driver.</p>
<p>Fixed a small bug that caused an error when receiving no results when using SQLResultSetRead.</p>

<h3>Changes in version 1.0.1 </h3>
<p>Improved error messages for Java errors that do not have messages but rather define the
error based on its class.</p>
<p>Fixed a bug that made SQLExpr data types expand very large within Java and often cause
OutOfMemory errors.  The fix involved converting SQLExpr data types into String within
Mathematica.  (SQLExpr data types are stored as strings within the database anyway.)</p>
<p>Added SQLResultSet functionality for customizing how result sets are processed.</p>
<p>Fixed a bug within jtds_sqlserver and jtds_sybase configuration files that caused problems
opening connections using these JDBC driver configurations.</p>
<p>Fixed a bug that left Java statements open. These need to be closed to free resources
for executing additional statements.</p>
<p>Fixed a bug in DatabaseExplorer that caused queries connected using the ODBC driver to fail
because of an unsupported feature.  The unsupported feature was the Timeout option.  The
Timeout option is no longer enabled when using the ODBC driver.  So ODBC queries by default
will now work, but without a timeout.</p>
<p>Fixed a bug in DatabaseExplorer that caused queries to fail if the Timeout option was not
checked.</p>
<p>Fixed DatabaseExplorer to no longer use a blank status message.  The blank message
caused DatabaseExplorer to redo the layout and looked jerky.</p>
<p>Fixed a problem that prevented DatabaseResources such JDBCDrivers and SQLConnections from
showing up in Linux in the functions JDBCDriverNames[], JDBCDrivers[], DataSourceNames[], and
DataSources if the resource had extra carriage returns at the beginning of the file.</p>

<h3>Changes in version 1.0.0 </h3>
Initial Release.

<h3>Known issues in this release </h3>
None.
