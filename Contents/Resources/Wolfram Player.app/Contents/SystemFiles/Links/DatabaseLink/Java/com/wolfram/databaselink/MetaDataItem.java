package com.wolfram.databaselink;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;

public enum MetaDataItem {
	AllProceduresAreCallable {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.allProceduresAreCallable());
		}
	},
	AllTablesAreSelectable {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.allTablesAreSelectable());
		}
	},
	CatalogSeparator {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getCatalogSeparator();
		}
	},
	CatalogTerm {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getCatalogTerm();
		}
	},
	DatabaseMajorVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getDatabaseMajorVersion());
		}
	},
	DatabaseMinorVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getDatabaseMinorVersion());
		}
	},
	DatabaseProductName {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getDatabaseProductName();
		}
	},
	DatabaseProductVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getDatabaseProductVersion();
		}
	},
	DataDefinitionCausesTransactionCommit {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.dataDefinitionCausesTransactionCommit());
		}
	},
	DataDefinitionIgnoredInTransactions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.dataDefinitionIgnoredInTransactions());
		}
	},
	DefaultTransactionIsolationLevel {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			String defaultTIL = "Undefined";

			switch (metaData.getDefaultTransactionIsolation()) {
			case Connection.TRANSACTION_READ_COMMITTED:
				defaultTIL = "ReadCommitted";
				break;
			case Connection.TRANSACTION_READ_UNCOMMITTED:
				defaultTIL = "ReadUncommitted";
				break;
			case Connection.TRANSACTION_REPEATABLE_READ:
				defaultTIL = "RepeatableRead";
				break;
			case Connection.TRANSACTION_SERIALIZABLE:
				defaultTIL = "Serializable";
				break;
			}

			return defaultTIL;
		}
	},
	DeletesAreDetectedForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.deletesAreDetected(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	DeletesAreDetectedForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.deletesAreDetected(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	DeletesAreDetectedForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.deletesAreDetected(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	DoesMaxRowSizeIncludeBlobs {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.doesMaxRowSizeIncludeBlobs());
		}
	},
	DriverMajorVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getDriverMajorVersion());
		}
	},
	DriverMinorVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getDriverMinorVersion());
		}
	},
	DriverName {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getDriverName();
		}
	},
	DriverVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getDriverVersion();
		}
	},
	ExtraNameCharacters {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getExtraNameCharacters();
		}
	},
	IdentifierQuoteString {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getIdentifierQuoteString();
		}
	},
	InsertsAreDetectedForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.insertsAreDetected(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	InsertsAreDetectedForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.insertsAreDetected(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	InsertsAreDetectedForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.insertsAreDetected(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	IsCatalogAtStartOfTableName {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.isCatalogAtStart());
		}
	},
	JDBCMajorVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getJDBCMajorVersion());
		}
	},
	JDBCMinorVersion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getJDBCMinorVersion());
		}
	},
	LocatorsUpdateCopy {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.locatorsUpdateCopy());
		}
	},
	MaxBinaryLiteralLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxBinaryLiteralLength());
		}
	},
	MaxCatalogNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxCatalogNameLength());
		}
	},
	MaxCharLiteralLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxCharLiteralLength());
		}
	},
	MaxColumnNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxColumnNameLength());
		}
	},
	MaxColumnsInGroupBy {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxColumnsInGroupBy());
		}
	},
	MaxColumnsInIndex {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxColumnsInIndex());
		}
	},
	MaxColumnsInOrderBy {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxColumnsInOrderBy());
		}
	},
	MaxColumnsInSelect {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxColumnsInSelect());
		}
	},
	MaxColumnsInTable {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxColumnsInTable());
		}
	},
	MaxConnections {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxConnections());
		}
	},
	MaxCursorNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxCursorNameLength());
		}
	},
	MaxIndexLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxIndexLength());
		}
	},
	MaxProcedureNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxProcedureNameLength());
		}
	},
	MaxRowSize {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxRowSize());
		}
	},
	MaxSchemaNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxSchemaNameLength());
		}
	},
	MaxStatementLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxStatementLength());
		}
	},
	MaxStatements {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxStatements());
		}
	},
	MaxTableNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxTableNameLength());
		}
	},
	MaxTablesInSelect {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxTablesInSelect());
		}
	},
	MaxUserNameLength {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Integer(metaData.getMaxUserNameLength());
		}
	},
	NullPlusNonNullIsNull {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.nullPlusNonNullIsNull());
		}
	},
	NullsAreSortedAtEnd {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.nullsAreSortedAtEnd());
		}
	},
	NullsAreSortedAtStart {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.nullsAreSortedAtStart());
		}
	},
	NullsAreSortedHight {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.nullsAreSortedHigh());
		}
	},
	NullsAreSortedLow {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.nullsAreSortedLow());
		}
	},
	NumericFunctions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getNumericFunctions();
		}
	},
	OthersDeletesAreVisibleForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.othersDeletesAreVisible(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	OthersDeletesAreVisibleForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.othersDeletesAreVisible(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	OthersDeletesAreVisibleForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.othersDeletesAreVisible(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	OthersInsertsAreVisibleForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.othersInsertsAreVisible(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	OthersInsertsAreVisibleForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.othersInsertsAreVisible(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	OthersInsertsAreVisibleForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.othersInsertsAreVisible(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	OthersUpdatesAreVisibleForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.othersUpdatesAreVisible(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	OthersUpdatesAreVisibleForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.othersUpdatesAreVisible(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	OthersUpdatesAreVisibleForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.othersUpdatesAreVisible(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	OwnDeletesAreVisibleForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.ownDeletesAreVisible(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	OwnDeletesAreVisibleForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.ownDeletesAreVisible(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	OwnDeletesAreVisibleForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.ownDeletesAreVisible(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	OwnInsertsAreVisibleForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.ownInsertsAreVisible(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	OwnInsertsAreVisibleForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.ownInsertsAreVisible(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	OwnInsertsAreVisibleForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.ownInsertsAreVisible(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	OwnUpdatesAreVisibleForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.ownUpdatesAreVisible(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	OwnUpdatesAreVisibleForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.ownUpdatesAreVisible(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	OwnUpdatesAreVisibleForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.ownUpdatesAreVisible(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	ProcedureTerm {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getProcedureTerm();
		}
	},
	ReadOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.isReadOnly());
		}
	},
	SchemaTerm {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getSchemaTerm();
		}
	},
	SearchStringEscape {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getSearchStringEscape();
		}
	},
	SQLKeywords {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getSQLKeywords();
		}
	},
	SQLStateType {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			String sqlStateType = "Undefined";

			switch (metaData.getDefaultTransactionIsolation()) {
			case DatabaseMetaData.sqlStateSQL99:
				sqlStateType = "SQL99";
				break;
			case DatabaseMetaData.sqlStateXOpen:
				sqlStateType = "XOpen";
				break;
			}

			return sqlStateType;
		}
	},
	StoresLowerCaseIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.storesLowerCaseIdentifiers());
		}
	},
	StoresLowerCaseQuotedIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.storesLowerCaseQuotedIdentifiers());
		}
	},
	StoresMixedCaseIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.storesMixedCaseIdentifiers());
		}
	},
	StoresMixedCaseQuotedIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.storesMixedCaseQuotedIdentifiers());
		}
	},
	StoresUpperCaseIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.storesUpperCaseIdentifiers());
		}
	},
	StoresUpperCaseQuotedIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.storesUpperCaseQuotedIdentifiers());
		}
	},
	StringFunctions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getStringFunctions();
		}
	},
	SupportsAlterTableWithAddColumn {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsAlterTableWithAddColumn());
		}
	},
	SupportsAlterTableWithDropColumn {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsAlterTableWithDropColumn());
		}
	},
	SupportsANSI92EntryLevelSQL {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsANSI92EntryLevelSQL());
		}
	},
	SupportsANSI92FullSQL {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsANSI92FullSQL());
		}
	},
	SupportsANSI92IntermediateSQL {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsANSI92IntermediateSQL());
		}
	},
	SupportsBatchUpdates {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsBatchUpdates());
		}
	},
	SupportsCatalogsInDataManipulation {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsCatalogsInDataManipulation());
		}
	},
	SupportsCatalogsInIndexDefinitions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsCatalogsInIndexDefinitions());
		}
	},
	SupportsCatalogsInPrivilegeDefinitions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsCatalogsInPrivilegeDefinitions());
		}
	},
	SupportsCatalogsInProcedureCalls {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsCatalogsInProcedureCalls());
		}
	},
	SupportsCatalogsInTableDefinitions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsCatalogsInTableDefinitions());
		}
	},
	SupportsColumnAliasing {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsColumnAliasing());
		}
	},
	SupportsConvert {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsConvert());
		}
	},
	SupportsCoreSQLGrammar {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsCoreSQLGrammar());
		}
	},
	SupportsCorrelatedSubqueries {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsCorrelatedSubqueries());
		}
	},
	SupportsDataDefinitionAndDataManipulationTransactions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsDataDefinitionAndDataManipulationTransactions());
		}
	},
	SupportsDataManipulationTransactionsOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsDataManipulationTransactionsOnly());
		}
	},
	SupportsDifferentTableCorrelationNames {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsDifferentTableCorrelationNames());
		}
	},
	SupportsExpressionsInOrderBy {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsExpressionsInOrderBy());
		}
	},
	SupportsExtendedSQLGrammar {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsExtendedSQLGrammar());
		}
	},
	SupportsForwardOnlyResultSetReadOnlyConcurrency {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsResultSetConcurrency(
							ResultSet.TYPE_FORWARD_ONLY,
							ResultSet.CONCUR_READ_ONLY));
		}
	},
	SupportsForwardOnlyResultSetType {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsResultSetType(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	SupportsForwardOnlyResultSetUpdatableConcurrency {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsResultSetConcurrency(
							ResultSet.TYPE_FORWARD_ONLY,
							ResultSet.CONCUR_UPDATABLE));
		}
	},
	SupportsFullOuterJoins {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsFullOuterJoins());
		}
	},
	SupportsGetGeneratedKeys {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsGetGeneratedKeys());
		}
	},
	SupportsGroupBy {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsGroupBy());
		}
	},
	SupportsGroupByBeyondSelect {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsGroupByBeyondSelect());
		}
	},
	SupportsGroupByUnrelated {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsGroupByUnrelated());
		}
	},
	SupportsIntegrityEnhancementFacility {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsIntegrityEnhancementFacility());
		}
	},
	SupportsLikeEscapeClause {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsLikeEscapeClause());
		}
	},
	SupportsLimitedOuterJoins {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsLimitedOuterJoins());
		}
	},
	SupportsMinimumSQLGrammar {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsMinimumSQLGrammar());
		}
	},
	SupportsMixedCaseIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsMixedCaseIdentifiers());
		}
	},
	SupportsMixedCaseQuotedIdentifiers {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsMixedCaseQuotedIdentifiers());
		}
	},
	SupportsMultipleOpenResults {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsMultipleOpenResults());
		}
	},
	SupportsMultipleResultSets {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsMultipleResultSets());
		}
	},
	SupportsMultipleTransactions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsMultipleTransactions());
		}
	},
	SupportsNamedParameters {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsNamedParameters());
		}
	},
	SupportsNonNullableColumns {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsNonNullableColumns());
		}
	},
	SupportsOpenCursorsAcrossCommit {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsOpenCursorsAcrossCommit());
		}
	},
	SupportsOpenCursorsAcrossRollback {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsOpenCursorsAcrossRollback());
		}
	},
	SupportsOpenStatementsAcrossCommit {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsOpenStatementsAcrossCommit());
		}
	},
	SupportsOpenStatementsAcrossRollback {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsOpenStatementsAcrossRollback());
		}
	},
	SupportsOrderByUnrelated {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsOrderByUnrelated());
		}
	},
	SupportsOuterJoins {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsOuterJoins());
		}
	},
	SupportsPositionedDelete {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsPositionedDelete());
		}
	},
	SupportsPositionedUpdate {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsPositionedUpdate());
		}
	},
	SupportsResultSetHoldCursorsOverCommitHoldability {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsResultSetHoldability(ResultSet.HOLD_CURSORS_OVER_COMMIT));
		}
	},
	SupportsResultSetCloseCursorsAtCommitHoldability {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsResultSetHoldability(ResultSet.CLOSE_CURSORS_AT_COMMIT));
		}
	},
	SupportsSavepoints {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSavepoints());
		}
	},
	SupportsSchemasInDataManipulation {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSchemasInDataManipulation());
		}
	},
	SupportsSchemasInIndexDefinitions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSchemasInIndexDefinitions());
		}
	},
	SupportsSchemasInPrivilegeDefinitions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsSchemasInPrivilegeDefinitions());
		}
	},
	SupportsSchemasInProcedureCalls {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSchemasInProcedureCalls());
		}
	},
	SupportsSchemasInTableDefinitions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSchemasInTableDefinitions());
		}
	},
	SupportsScrollInsensitiveResultSetReadOnlyConcurrency {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsResultSetConcurrency(
					ResultSet.TYPE_SCROLL_INSENSITIVE,
					ResultSet.CONCUR_READ_ONLY));
		}
	},
	SupportsScrollInsensitiveResultSetType {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsResultSetType(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	SupportsScrollInsensitiveResultSetUpdatableConcurrency {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsResultSetConcurrency(
					ResultSet.TYPE_SCROLL_INSENSITIVE,
					ResultSet.CONCUR_UPDATABLE));
		}
	},
	SupportsScrollSensitiveResultSetReadOnlyConcurrency {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsResultSetConcurrency(
					ResultSet.TYPE_SCROLL_SENSITIVE,
					ResultSet.CONCUR_READ_ONLY));
		}
	},
	SupportsScrollSensitiveResultSetType {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.supportsResultSetType(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	SupportsScrollSensitiveResultSetUpdatableConcurrency {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsResultSetConcurrency(
					ResultSet.TYPE_SCROLL_SENSITIVE,
					ResultSet.CONCUR_UPDATABLE));
		}
	},
	SupportsSelectForUpdate {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSelectForUpdate());
		}
	},
	SupportsStatementPooling {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsStatementPooling());
		}
	},
	SupportsStoredProcedures {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsStoredProcedures());
		}
	},
	SupportsSubqueriesInComparisons {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSubqueriesInComparisons());
		}
	},
	SupportsSubqueriesInExists {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSubqueriesInExists());
		}
	},
	SupportsSubqueriesInIns {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSubqueriesInIns());
		}
	},
	SupportsSubqueriesInQuantifieds {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsSubqueriesInQuantifieds());
		}
	},
	SupportsTableCorrelationNames {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsTableCorrelationNames());
		}
	},
	SupportsReadCommitedTransactionIsolationLevel {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsTransactionIsolationLevel(Connection.TRANSACTION_READ_COMMITTED));
		}
	},
	SupportsReadUncommitedTransactionIsolationLevel {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsTransactionIsolationLevel(Connection.TRANSACTION_READ_UNCOMMITTED));
		}
	},
	SupportsRepeatableReadTransactionIsolationLevel {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsTransactionIsolationLevel(Connection.TRANSACTION_REPEATABLE_READ));
		}
	},
	SupportsSerializableTransactionIsolationLevel {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(
					metaData
					.supportsTransactionIsolationLevel(Connection.TRANSACTION_SERIALIZABLE));
		}
	},
	SupportsTransactions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsTransactions());
		}
	},
	SupportsUnion {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsUnion());
		}
	},
	SupportsUnionAll {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.supportsUnionAll());
		}
	},
	SystemFunctions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getSystemFunctions();
		}
	},
	TimeDateFunctions {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getTimeDateFunctions();
		}
	},
	UpdatesAreDetectedForForwardOnly {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.updatesAreDetected(ResultSet.TYPE_FORWARD_ONLY));
		}
	},
	UpdatesAreDetectedForScrollInsensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.updatesAreDetected(ResultSet.TYPE_SCROLL_INSENSITIVE));
		}
	},
	UpdatesAreDetectedForScrollSensitive {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData
					.updatesAreDetected(ResultSet.TYPE_SCROLL_SENSITIVE));
		}
	},
	URL {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getURL();
		}
	},
	UserName {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return metaData.getUserName();
		}
	},
	UsesLocalFilePerTable {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.usesLocalFilePerTable());
		}
	},
	UsesLocalFiles {
		Object getValue(DatabaseMetaData metaData) throws Exception {
			return new Boolean(metaData.usesLocalFiles());
		}
	};

	abstract Object getValue(DatabaseMetaData metaData) throws Exception;
}
