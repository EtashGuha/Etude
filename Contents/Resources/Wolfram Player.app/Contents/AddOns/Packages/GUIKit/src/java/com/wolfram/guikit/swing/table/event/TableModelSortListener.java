/*
 * @(#)TableModelSortListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing.table.event;

/**
 * TableModelSortListener defines the interface for an object that listens
 * to sort changes in a TableModel.
 *
 * @version 1.0
 */

public interface TableModelSortListener extends java.util.EventListener
{
    /**
     * This fine grain notification tells listeners the exact column and
     * reordering of rows that sorted.
     */
    public void tableSorted(TableModelSortEvent e);
}

