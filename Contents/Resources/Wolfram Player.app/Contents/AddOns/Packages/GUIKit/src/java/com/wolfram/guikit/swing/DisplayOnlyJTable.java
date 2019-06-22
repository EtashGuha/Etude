/*
 * @(#)DisplayOnlyJTable
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.Vector;

import javax.swing.table.TableColumnModel;

import javax.swing.ListSelectionModel;

import com.wolfram.guikit.swing.table.SortTableModel;

/**
 * DisplayOnlyJTable extends ExprAccessibleJTable
 */
public class DisplayOnlyJTable extends ExprAccessibleJTable {
  
  private static final long serialVersionUID = -1187987975456188148L;
  
  private boolean allowEdits = false;
  
	public DisplayOnlyJTable() {
		super();
		}
	public DisplayOnlyJTable(int numRows, int numColumns) {
		super(numRows, numColumns);
		}
	public DisplayOnlyJTable(final Object[][] rowData, final Object[] columnNames) {
		super(rowData, columnNames);
		}
	public DisplayOnlyJTable(Vector rowData, Vector columnNames) {
		super(rowData, columnNames);
		}
	public DisplayOnlyJTable(SortTableModel dm) {
		super(dm);
		}
	public DisplayOnlyJTable(SortTableModel dm, TableColumnModel cm) {
		super(dm, cm);
		}
	public DisplayOnlyJTable(SortTableModel dm, TableColumnModel cm, ListSelectionModel sm) {
		super(dm, cm, sm);
		}
							
  public void setAllowEdits(boolean newVal) {
    allowEdits = newVal;
    }				
    					
	public boolean isCellEditable(int row, int col) {
    return allowEdits;
    }
    
  }
