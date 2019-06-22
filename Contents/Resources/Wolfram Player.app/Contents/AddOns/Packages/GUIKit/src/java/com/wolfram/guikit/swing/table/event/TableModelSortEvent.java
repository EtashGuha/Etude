/*
 * @(#)TableModelSortEvent.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing.table.event;

import javax.swing.table.*;

/**
 * TableModelSortEvent
 *
 * @version 1.0
 */
public class TableModelSortEvent extends java.util.EventObject {
  
  private static final long serialVersionUID = -1287989975456798998L;
  
  protected int	column;
  protected int[] ordering;
  
  public TableModelSortEvent(TableModel source, int column, int[] ordering) {
    super(source);
    this.column = column;
    this.ordering = ordering;
    }

//
// Querying Methods
//

  public int getColumn() { return column; };

  public int[] getOrdering() { return ordering; }
}
