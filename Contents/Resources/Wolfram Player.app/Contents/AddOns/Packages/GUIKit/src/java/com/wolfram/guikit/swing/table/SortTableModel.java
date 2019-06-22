/*
 * @(#)SortTableModel.java 1.14 03/01/23
 * 
 * Based on JavaPro article
 * http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/
 * Created by Claude Duguay
 * Copyright (c) 2002
 */
package com.wolfram.guikit.swing.table;

import javax.swing.table.TableModel;

import com.wolfram.guikit.swing.table.event.TableModelSortListener;

/**
 * SortTableModel
 */
public interface SortTableModel extends TableModel {
  
  public boolean getSortableByDefault();
  public void setSortableByDefault(boolean val);

  public void setColumnSortable(Object r);
  public Object getColumnSortable();

  public boolean isSortable(int col);
  
  public void sortColumn(int col, boolean ascending);
  
  public void addTableModelSortListener(TableModelSortListener l);
  public void removeTableModelSortListener(TableModelSortListener l);
    
}

