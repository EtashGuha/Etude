/*
 * @(#)DefaultEdgeRenderer.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.graph;

import java.awt.Image;

import com.wolfram.guikit.diva.ImageLabelFigure;
import com.wolfram.jlink.Expr;

import diva.graph.*;
import diva.util.PropertyContainer;
import diva.canvas.Figure;
import diva.canvas.Site;
import diva.canvas.connector.AbstractConnector;
import diva.canvas.connector.ArcConnector;
import diva.canvas.connector.Connector;
import diva.canvas.connector.Arrowhead;
import diva.canvas.connector.StraightConnector;
import diva.canvas.toolbox.LabelFigure;

/**
 * DefaultEdgeRenderer is a default implementation of the EdgeRenderer interface.
 *
 * @version $Revision: 1.2 $
 */
public class DefaultEdgeRenderer implements EdgeRenderer {    
	
	 public DefaultEdgeRenderer(GraphController controller) {
	   }
	   
    /**
     * Render a visual representation of the given edge.
     */
    public Connector render(Object edge, Site tailSite, Site headSite) {
        AbstractConnector c;
        Figure tf = tailSite.getFigure();
        Figure hf = headSite.getFigure();
			  PropertyContainer p = (PropertyContainer)edge;
			  
        //if the edge is a self loop, create an ArcConnector instead

        if((tf != null) && (hf != null) && (tf == hf)){
          c = new ArcConnector(tailSite, headSite);
          }
        else {
        	Object type = p.getProperty("connectorType");
        	if (type != null && type.equals("Straight")) {
						c = new StraightConnector(tailSite, headSite);
        	  }
        	else {
						c = new ArcConnector(tailSite, headSite);
        	  }
          }
          
			  Object directed = p.getProperty("directed");
			  if (directed != null && directed.equals(Boolean.TRUE)) {
				  Arrowhead arrow = new Arrowhead(headSite.getX(), headSite.getY(), headSite.getNormal());
				  c.setHeadEnd(arrow);
				  }
				  
        Object imageObject = p.getProperty("image");
        if (imageObject != null && imageObject instanceof Image) {
          c.setLabelFigure( new ImageLabelFigure((Image)imageObject));
          }
        else {
          Object exprObject = p.getProperty("expr");
          if (exprObject != null && exprObject instanceof Expr) {
            c.setLabelFigure( new LabelFigure(((Expr)exprObject).toString()));
            }
          }
        
			  //Object scm = model.getSemanticObject(edge);
				//c.setToolTipText( scm != null ? scm.toString() : c.toString());
	
        return c;
      }
}



