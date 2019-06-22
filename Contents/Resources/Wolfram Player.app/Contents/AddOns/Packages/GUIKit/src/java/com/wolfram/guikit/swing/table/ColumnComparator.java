/*
 * @(#)ColumnComparator.java 1.14 03/01/23
 */
package com.wolfram.guikit.swing.table;

import java.util.Comparator;
import java.util.Vector;

/**
 * ColumnComparator
 * 
 * Based on JavaPro article
 * http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/
 * Created by Claude Duguay
 * Copyright (c) 2002
 */
public class ColumnComparator implements Comparator {
  
  protected int index;
  protected boolean ascending;
  
  public ColumnComparator(int index, boolean ascending) {
    this.index = index;
    this.ascending = ascending;
    }
  
  public int compare(Object one, Object two) {
    if (one instanceof Vector && two instanceof Vector) {
      Vector vOne = (Vector)one;
      Vector vTwo = (Vector)two;
      Object oOne = vOne.elementAt(index);
      Object oTwo = vTwo.elementAt(index);
      // By default for String we use case-insensitive sorting
      if (oOne instanceof String && oTwo instanceof String) {
        if (ascending) return ((String)oOne).compareToIgnoreCase((String)oTwo);
        else return ((String)oTwo).compareToIgnoreCase((String)oOne);
        }
      else if (oOne instanceof Comparable && oTwo instanceof Comparable) {
        if (ascending) return ((Comparable)oOne).compareTo((Comparable)oTwo);
        else return ((Comparable)oTwo).compareTo((Comparable)oOne);
        }
      }
    return 1;
    }
    
}

