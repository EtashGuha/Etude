/*
 * @(#)DefaultEdgeController.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.graph;

import java.awt.event.InputEvent;

import diva.canvas.Figure;
import diva.canvas.FigureLayer;
import diva.canvas.Site;
import diva.canvas.connector.AutonomousSite;
import diva.canvas.connector.Connector;
import diva.canvas.connector.ConnectorEvent;
import diva.canvas.event.MouseFilter;
import diva.canvas.interactor.BasicSelectionRenderer;
import diva.canvas.interactor.Manipulator;

import diva.graph.BasicEdgeController;
import diva.graph.EdgeInteractor;
import diva.graph.GraphController;
import diva.graph.GraphException;
import diva.graph.MutableGraphModel;

/**
 * DefaultEdgeController
 *
 * @version	$Revision: 1.2 $
 */
public class DefaultEdgeController extends BasicEdgeController {

  public DefaultEdgeController(GraphController controller) {
    super(controller);  
    // We have to find and reset default mouse filter to one we use for DefaultGraphController
    BasicSelectionRenderer renderer = (BasicSelectionRenderer)((EdgeInteractor)getEdgeInteractor()).getSelectionRenderer();
    ((Manipulator)renderer.getDecorator()).setHandleFilter(
       new MouseFilter(InputEvent.BUTTON1_MASK|InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK,0,0));
    }
   
   /** Add an edge to this graph editor and render it
   * from the given tail node to an autonomous site at the
   * given location. Give the new edge the given semanticObject.
   * The "end" flag is either HEAD_END
   * or TAIL_END, from diva.canvas.connector.ConnectorEvent.
   * @return The new edge.
   * @exception GraphException If the connector target cannot return a 
   * valid site on the node's figure.
   */
  public void addEdge(Object edge, Object node, int end, double x, double y) {
    MutableGraphModel model = (MutableGraphModel) getController().getGraphModel();
    Figure nf = getController().getFigure(node);
    FigureLayer layer = getController().getGraphPane().getForegroundLayer();
    Site headSite, tailSite;

    // Temporary sites.  One of these will get blown away later.
    headSite = new AutonomousSite(layer, x, y);          
    tailSite = new AutonomousSite(layer, x, y);

    // Render the edge.
    Connector c = render(edge, layer, tailSite, headSite);

    try {
      //Attach the appropriate end of the edge to the node.
      if (end == ConnectorEvent.TAIL_END) {
        tailSite = getConnectorTarget().getTailSite(c, nf, x, y);
        if(tailSite == null) {
          throw new RuntimeException("Invalid connector target: " + 
          "no valid site found for tail of new connector.");
          }
        model.setEdgeTail(getController(), edge, node);
        c.setTailSite(tailSite);
        } 
      else {
        headSite = getConnectorTarget().getHeadSite(c, nf, x, y);    
        if(headSite == null) {
          throw new RuntimeException("Invalid connector target: " + 
          "no valid site found for head of new connector.");
          }
        model.setEdgeHead(getController(), edge, node);
        c.setHeadSite(headSite);
        }
      } 
    catch (GraphException ex) {
      // If an error happened then blow away the edge, and rethrow
      // the exception
      removeEdge(edge);
      throw ex;
      }
    }
 
  public void addEdge(Object edge, double x1, double y1, double x2, double y2) {
    FigureLayer layer = getController().getGraphPane().getForegroundLayer();
    Site headSite, tailSite;

    headSite = new AutonomousSite(layer, x2, y2);          
    tailSite = new AutonomousSite(layer, x1, y1);

    // Render the edge.
    render(edge, layer, tailSite, headSite);
    }
    
}



