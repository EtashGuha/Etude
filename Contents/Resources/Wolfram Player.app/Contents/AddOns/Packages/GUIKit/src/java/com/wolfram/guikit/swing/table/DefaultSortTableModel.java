/*
 * @(#)DefaultSortTableModel.java 1.14 03/01/23
 */
package com.wolfram.guikit.swing.table;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

import javax.swing.event.EventListenerList;
import javax.swing.table.DefaultTableModel;

import com.wolfram.guikit.swing.ItemListModel;
import com.wolfram.guikit.swing.table.event.TableModelSortEvent;
import com.wolfram.guikit.swing.table.event.TableModelSortListener;

/**
 * DefaultSortTableModel
 * 
 * Based on JavaPro article
 * http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/
 * Created by Claude Duguay
 * Copyright (c) 2002
 */
public class DefaultSortTableModel extends DefaultTableModel implements SortTableModel {
  
  private static final long serialVersionUID = -1387987935453788948L;
    
  protected EventListenerList sortListeners = null;
  
  private boolean sortableByDefault = false;
  private Object[] columnSortable = null;
  
  protected int[] ordering = null;
  
  public DefaultSortTableModel() {}
  
  public DefaultSortTableModel(int rows, int cols) {
    super(rows, cols);
    }
  
  public DefaultSortTableModel(Object[][] data, Object[] names) {
    super(data, names);
    }
  
  public DefaultSortTableModel(Object[] names, int rows) {
    super(names, rows);
    }
  
  public DefaultSortTableModel(Vector names, int rows) {
    super(names, rows);
    }
  
  public DefaultSortTableModel(Vector data, Vector names) {
    super(data, names);
    }
  
  public boolean getSortableByDefault() {return sortableByDefault;}
  public void setSortableByDefault(boolean val) {
    sortableByDefault = val;
    }
  
  public void setColumnSortable(Object r) {
    columnSortable = ItemListModel.convertToArray(r);
    if (columnSortable == null && r != null && r instanceof Boolean) {
      columnSortable = new Boolean[]{(Boolean)r};
      }
    }
  public Object getColumnSortable() {
    return columnSortable;
    }
  
  public boolean isSortable(int col) {
    Object columnObject = null;
    if (columnSortable != null) {
      if (columnSortable instanceof Object[]) {
        Object[] columnArray = (Object[])columnSortable;
        if (col < columnArray.length) columnObject = columnArray[col];
        else if (columnArray.length >= 1) columnObject = columnArray[0];
        }
      else columnObject = columnSortable;
      }
    if (columnObject != null) {
      if (columnObject instanceof Boolean) return ((Boolean)columnObject).booleanValue();
      }
    return sortableByDefault;
    }
    
  public int[] getOrdering() { return ordering; }
  
  public void sortColumn(int col, boolean ascending) {

    Comparator c = new ColumnComparator(col, ascending);
    List list = getDataVector();
    
    Object a[] = list.toArray();
    Arrays.sort(a, c);
    
    int[] newOrdering = new int[a.length];
    boolean reordered = false;
    
    for (int j=0; j < a.length; j++) {
      int newIndex = list.indexOf(a[j]);
      newOrdering[j] = newIndex;
      if (newIndex != j) reordered = true;
      }
    
    if (reordered) {
      ListIterator i = list.listIterator();
      for (int j=0; j < a.length; j++) {
        i.next();
        i.set(a[j]);
        }
       
      ordering = newOrdering;
      fireSortedEvent(col, newOrdering);
      }
    }


  /**
   * Adds the specified ParameterListener to receive ParameterEvents.
   * <p>
   * Use this method to register a ParameterListener object to receive
   * notifications when parameter changes occur
   *
   * @param l the ParameterListener to register
   * @see #removeParameterListener(ParameterListener)
   */
  public void addTableModelSortListener(TableModelSortListener l) {
    if (sortListeners == null) sortListeners = new EventListenerList();
    if ( l != null ) {
      sortListeners.add( TableModelSortListener.class, l );
      }
    }

  /**
   * Removes the specified ParameterListener object so that it no longer receives
   * ParameterEvents.
   *
   * @param l the ParameterListener to register
   * @see #addParameterListener(ParameterListener)
   */
  public void removeTableModelSortListener(TableModelSortListener l) {
    if (sortListeners == null) return;
    if ( l != null ) {
      sortListeners.remove( TableModelSortListener.class, l );
      }
    }

  public void fireSortedEvent(int col, int[] ordering) {
    if (sortListeners == null) return;
    Object[] lsns = sortListeners.getListenerList();
    TableModelSortEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == TableModelSortListener.class ) {
        if (e == null)
          e = new TableModelSortEvent( this, col, ordering);
        ((TableModelSortListener)lsns[i+1]).tableSorted( e );
        }
      }
    }

}

