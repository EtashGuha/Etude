/*
 * @(#)CompoundMouseFilter.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.diva;

import java.awt.event.MouseEvent;
import diva.canvas.event.MouseFilter;

/**
 * CompoundMouseFilter is a subclass of MouseFilter that
 * wraps a list of multiple MouseFilters that allows any one
 * to return true for an accept.
 * 
 * @version $Revision: 1.1 $
 */
public class CompoundMouseFilter extends MouseFilter {

    private MouseFilter[] filters = null;
   
    public CompoundMouseFilter(MouseFilter[] filters) {
      super(0);
      this.filters = filters;
      }

    /**
     * Test whether the given MouseEvent passes the filters.
     */
    public boolean accept(MouseEvent event) {
      for (int i = 0; i < filters.length; ++i) {
        if (filters[i].accept(event))
          return true;
        }
      return false;
      }

    /** Print a useful description of the mouse filters.
     */
    public String toString () {
      StringBuffer result = new StringBuffer();
      result.append(getClass().getName() + "@" + Integer.toHexString(hashCode()));
      result.append("; Length " + (filters != null ? filters.length : 0) + " ");
      for (int i = 0; i < filters.length; ++i) {
        result.append("; ");
        result.append(filters[i].toString());
        }
      return result.toString();
      }
}



