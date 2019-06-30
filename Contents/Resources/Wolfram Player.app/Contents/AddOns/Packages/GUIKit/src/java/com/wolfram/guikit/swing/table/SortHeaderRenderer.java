/*
 * @(#)SortHeaderRenderer.java 1.14 03/01/23
 */
package com.wolfram.guikit.swing.table;

import java.awt.Component;

import javax.swing.UIManager;
import javax.swing.Icon;
import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.JTableHeader;

/**
 * SortHeaderRenderer
 * 
 * Based on JavaPro article
 * http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/
 * Created by Claude Duguay
 * Copyright (c) 2002
 */
public class SortHeaderRenderer extends DefaultTableCellRenderer {
  
  private static final long serialVersionUID = -1277987975457788748L;
  
  public static Icon NONSORTED =
    new SortArrowIcon(SortArrowIcon.NONE);
  public static Icon ASCENDING =
    new SortArrowIcon(SortArrowIcon.ASCENDING);
  public static Icon DECENDING =
    new SortArrowIcon(SortArrowIcon.DECENDING);
  
  public SortHeaderRenderer()
  {
    setHorizontalTextPosition(LEFT);
    setHorizontalAlignment(CENTER);
  }
  
  public Component getTableCellRendererComponent(
    JTable table, Object value, boolean isSelected,
    boolean hasFocus, int row, int col)
  {
    int index = -1;
    boolean ascending = true;
    if (table instanceof JSortTable)
    {
      JSortTable sortTable = (JSortTable)table;
      index = sortTable.getSortedColumnIndex();
      ascending = sortTable.isSortedColumnAscending();
    }
    if (table != null)
    {
      JTableHeader header = table.getTableHeader();
      if (header != null)
      {
        setForeground(header.getForeground());
        setBackground(header.getBackground());
        setFont(header.getFont());
      }
    }
    Icon icon = ascending ? ASCENDING : DECENDING;
    setIcon(col == index ? icon : NONSORTED);
    setText((value == null) ? "" : value.toString());
    setBorder(UIManager.getBorder("TableHeader.cellBorder"));
    return this;
  }
}

