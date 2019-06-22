/*
 * @(#)DefaultConnectorTarget.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.graph;

import diva.canvas.Figure;

import diva.canvas.connector.Connector;
import diva.canvas.connector.PerimeterTarget;

/** 
 * DefaultConnectorTarget returns sites on the perimeter of a figure.
 *
 * @version $Revision: 1.1 $
 */
public class DefaultConnectorTarget extends PerimeterTarget  {

    /** Return true if the given connector can be connected to the given
     * figure.  In this base class return true if the tail of the connector
     * is not attached to the same figure.
     */
    public boolean acceptHead(Connector c, Figure f) {
      return true;
    }

    /** Return true if the given connector can be connected to the given
     * figure.  In this base class return true if the head of the connector
     * is not attached to the same figure.
     */
    public boolean acceptTail(Connector c, Figure f) {
      return true;
    }
    
}



