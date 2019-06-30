/*
 * @(#)TrekPane.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.Figure;
import diva.canvas.FigureDecorator;
import diva.canvas.CanvasUtilities;
import diva.canvas.FigureLayer;
import diva.canvas.GraphicsPane;
import diva.canvas.JCanvas;
import diva.canvas.event.LayerEvent;
import diva.canvas.interactor.Interactor;
import diva.canvas.interactor.SelectionModel;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.geom.*;

import javax.swing.event.EventListenerList;

import com.wolfram.guikit.diva.AxesFigure;

import com.wolfram.guikit.trek.Trek;
import com.wolfram.guikit.trek.TrekToolBar;
import com.wolfram.guikit.trek.event.TrekEvent;
import com.wolfram.guikit.trek.event.TrekListener;

/**
 * TrekPane is a diva pane that illustrates treks.
 *
 * @version $Revision: 1.5 $
 */
public class TrekPane extends GraphicsPane {

  public static final int LINE = TrekFigure.LINE;
  public static final int POINTS = TrekFigure.POINTS;
  
  public static NumberFormat defaultDecimalFormat = NumberFormat.getInstance();

  protected EventListenerList listeners = null;
  
  private Hashtable trekFigureHash = new Hashtable();
  private Hashtable trekHash = new Hashtable();

  private TrekToolBar trekToolBar;
  
  private static long uniqueID = 0;
  
  /** The controller */
  protected TrekController controller;

	protected double defaultIndependentRange[] = new double[2];
	protected double defaultOriginIndependent = 0.0;
	
  // Conversion from Mathematica to Diva coordinate space

  double xmin = 0.0; 
  double xmax = 0.0;
  double ymin = 100.0;
  double ymax = 100.0;

  double xscale = 1.0; 
  double yscale = 1.0;
  
  private AxesFigure axesFigure = null;
  
  private int defaultDisplayMode = TrekFigure.LINE;
  
  public TrekPane(TrekController controller) {
    super();
    
    // default is 0.5 but lets make this a tad larger
    // for tooltipping
    getForegroundLayer().setPickHalo(1.0);
    
    setTrekController(controller);
    
    AffineTransform t = new AffineTransform();
    t.setToIdentity();
    setTransform(t);
    
    // Off til working correctly
    axesFigure = new AxesFigure(null, null);
    FigureLayer layer = (FigureLayer)getBackgroundLayer();
    layer.setVisible(true);
    layer.add(axesFigure);
    
    }

  public int getDefaultDisplayMode() {return defaultDisplayMode;}
  public void setDefaultDisplayMode(int newMode) {
    defaultDisplayMode = newMode;
    if (trekToolBar != null) trekToolBar.setDefaultDisplayMode(defaultDisplayMode);
    }
    
	public void updateCursor() {
		((TrekCanvas)getCanvas()).updateCursor();
		}
	
  static {
    defaultDecimalFormat.setMaximumFractionDigits(8);
    defaultDecimalFormat.setGroupingUsed(false);
    }
  
  public Rectangle2D getTrekFigureBounds() {
    Rectangle2D resultBounds = null;
    Iterator it = trekFigureHash.values().iterator();
    while(it.hasNext()) {
      Rectangle2D b = ((TrekFigure)it.next()).getPointsFigure().getBounds();
      if (resultBounds == null) {
        resultBounds = new Rectangle2D.Double(b.getX(), b.getY(), b.getWidth(), b.getHeight());
        }
      else {
        Rectangle2D.union(resultBounds, b, resultBounds);
        }
      }
    return resultBounds;
    }
  
  // Zoom to Fit preserves the panel's current aspect ratio
  // while Scale to Fit will change x and y scales independently
  
  public void zoomToFit() {
		Dimension visSize = getVisibleSize();
		
		Rectangle2D totalBounds = getTrekFigureBounds();
		if (totalBounds == null || totalBounds.getWidth() == 0.0 || totalBounds.getHeight() == 0.0) return;
		
		double currAspect = totalBounds.getHeight()/totalBounds.getWidth();
		if (currAspect == Double.NaN) return;
		
		double zoomAmount = 1.0;
		if (currAspect >= 1.0) zoomAmount = visSize.getHeight()/totalBounds.getHeight();
	  else zoomAmount = visSize.getWidth()/totalBounds.getWidth();
		if (zoomAmount == Double.POSITIVE_INFINITY || zoomAmount == Double.NEGATIVE_INFINITY) return;
    
		double newCenterX = getTrekCoordinateX(totalBounds.getCenterX());
		double newCenterY = getTrekCoordinateY(totalBounds.getCenterY());
		addScaleFactor(zoomAmount, zoomAmount, newCenterX, newCenterY);
  	}
  
  public void scaleToFit() {
    Dimension visSize = getVisibleSize();
    
    Rectangle2D totalBounds = getTrekFigureBounds();
    if (totalBounds == null || totalBounds.getWidth() == 0.0 || totalBounds.getHeight() == 0.0) return;
    
    double xZoomAmount = 1.0;
    double yZoomAmount = 1.0;
    
    yZoomAmount = visSize.getHeight()/totalBounds.getHeight();
    xZoomAmount = visSize.getWidth()/totalBounds.getWidth();
    if (xZoomAmount == Double.POSITIVE_INFINITY || xZoomAmount == Double.NEGATIVE_INFINITY ||
      yZoomAmount == Double.POSITIVE_INFINITY || yZoomAmount == Double.NEGATIVE_INFINITY) return;
    
    double newCenterX = getTrekCoordinateX(totalBounds.getCenterX());
    double newCenterY = getTrekCoordinateY(totalBounds.getCenterY());
    addScaleFactor(xZoomAmount, yZoomAmount, newCenterX, newCenterY);
    }
    
	/** Return the size of the visible part of a canvas, in canvas
	 *  coordinates.
	 */
	public Dimension getVisibleSize() {
    JCanvas c = getCanvas();
    if (c == null) return null;
    else return c.getSize();
		}
		
  // This exists so that we can update figures that need
  // to know the canvas bounds, such as AxesFigures
  public void canvasReshaped(int x, int y, int w, int h) {
    xmax = xmin + w/xscale;
    ymin = ymax - h/yscale;
     
    if (axesFigure != null) 
       axesFigure.canvasReshaped(x,y,w,h);
    }
  
  /**
   * Get the sketch controller that controls
   * the behavior of this pane.
   */
  public TrekController getTrekController() {
    return controller;
    }

  public TrekToolBar getTrekToolBar() {return trekToolBar;}
  public void setTrekToolBar(TrekToolBar toolBar) {
    trekToolBar = toolBar;
    }
  
  /**
   * Set the sketch controller that controls
   * the behavior of this pane.
   */
  private void setTrekController (TrekController c) {
    controller = c;
    controller.setTrekPane(this);
    }

  public Trek getTrek(String key) {
   return (Trek)trekHash.get(key);
   }
  public Trek[] getTreks() {
    String[] keys = getTrekKeys();
    Trek[] treks = new Trek[keys.length];
    for (int i = 0; i < treks.length; ++i) {
      treks[i] = getTrek(keys[i]);
      }
    return treks;
    }
  
  public TrekFigure[] getTrekFigures() {
    String[] keys = getTrekKeys();
    TrekFigure[] trekFigs = new TrekFigure[keys.length];
    for (int i = 0; i < trekFigs.length; ++i) {
      trekFigs[i] = getTrekFigure(keys[i]);
      }
    return trekFigs;
    }
  public TrekFigure getTrekFigure(String key) {
   return (TrekFigure)trekFigureHash.get(key);
   }
   
  public String[] getTrekKeys() {
    return (String[])trekHash.keySet().toArray(new String[0]);
    }

  public double[] getTrekOrigin(String trekID) {
    Trek t = getTrek(trekID);
    if (t != null) {
      return t.getOrigin();
      }
    return null;
    }

	public double[] getDefaultIndependentRange() {return defaultIndependentRange;}
	public void setDefaultIndependentRange(double[] newRange) {
		defaultIndependentRange[0] = newRange[0];
		defaultIndependentRange[1] = newRange[1];
		}
	
  public void setDefaultIndependentRangeMin(double newVal) {
    defaultIndependentRange[0] = newVal;
    }
  public void setDefaultIndependentRangeMax(double newVal) {
    defaultIndependentRange[1] = newVal;
    }
    
	public double getDefaultOriginIndependent() {return defaultOriginIndependent;}
	public void setDefaultOriginIndependent(double newVal) {
		defaultOriginIndependent = newVal;
		}
		
	public double[] getTrekIndependentRange(String trekID) {
		Trek t = getTrek(trekID);
		if (t != null) {
			return t.getIndependentRange();
			}
		return getDefaultIndependentRange();
		}
		
	public double getTrekOriginIndependent(String trekID) {
		Trek t = getTrek(trekID);
		if (t != null) {
			return t.getOriginIndependent();
			}
		return getDefaultOriginIndependent();
		}
		
  public void removeTrek(String trekID) {
    Trek t = getTrek(trekID);
    if (t != null) {
      // Need to tell trekPane to remove the figure
      removeTrekFigure(trekID);
      trekHash.remove(trekID);
      }
    }

  public double[][] getTrekPoints(String trekID) {
    Trek t = getTrek(trekID);
		if (t != null) return t.getPoints();
		else return null;
    }
  
  public void setTrekPoints(String trekID, double[] origin, double indepZero, double[] independRange, double points[][]) {
    String useID = trekID;
    if (useID == null) {
      Object selectionFigure = controller.getSelectionModel().getFirstSelection();
      if (selectionFigure != null && selectionFigure instanceof Figure) {
        useID = getTrekIdFromTarget((Figure)selectionFigure);
        if (useID == null) return;
        }
      }
    Trek t = getTrek(useID);
    if (t == null) {
      t = new Trek(useID, origin, indepZero, independRange);
      trekHash.put(t.getKey(), t);
      }
    else {
      t.setOrigin(origin);
      t.setOriginIndependent(indepZero);
      t.setIndependentRange(independRange);
      }
    t.setPoints(points);
    setTrekFigurePoints(useID, origin, indepZero, independRange, points);
    }

  /** Get the selection interactor
   */
  public Interactor getSelectionInteractor() {
    return controller.getSelectionInteractor();
    }
  
  protected void setTrekFigurePoints(String trekID, double[] origin, double originIndep, double[] independentRange, double points[][]) {
    TrekFigure fig = getTrekFigure(trekID);
    if (fig != null) {
      fig.update(origin, points, xmin, xscale, ymax, yscale);
      }
    else {
      controller.getSelectionModel().clearSelection();
      fig = new TrekFigure(trekID);
      Color newColor = fig.createRandomColor();
      fig.setColor(newColor);
      fig.init(origin, points, xmin, xscale, ymax, yscale);
      fig.setDisplayMode(getDefaultDisplayMode());
      fig.getSelectionFigure().setInteractor( getSelectionInteractor());
      getForegroundLayer().add(fig);
      trekFigureHash.put(trekID, fig);
      
      controller.getSelectionModel().addSelection(fig.getSelectionFigure());
      getCanvas().requestFocus();
      }

    if (fig != null) {
      fig.setToolTipText("(" + defaultDecimalFormat.format(origin[0]) + "," + 
        defaultDecimalFormat.format(origin[1]) + ")");
      }
    }

  public String getTrekIdFromTarget(Figure f) {
    if (f == null) return null;
    Figure useFig = f;
    if (useFig instanceof FigureDecorator) {
      useFig = ((FigureDecorator)useFig).getDecoratedFigure();
      }
    if (useFig != null) return (String)useFig.getUserObject();
    else return null;
    }

  public double[] getTargetCoordinate(Figure f) {
    Point2D p = getTargetCenter(f);
    return new double[]{getTrekCoordinateX(p.getX()), getTrekCoordinateY(p.getY())};
    }
  
  public Point2D getTargetCenter(Figure f) {
    Figure useFig = f;
    if (useFig instanceof FigureDecorator) {
      useFig = ((FigureDecorator)useFig).getDecoratedFigure();
      }
    return CanvasUtilities.getCenterPoint(useFig);
    }

  public void removeTrekFigure(String trekID) {
    TrekFigure fig = getTrekFigure(trekID);
    if (fig != null) {
      getForegroundLayer().remove(fig);
      trekFigureHash.remove(trekID);
      }
    }

  public void addScaleFactor(double newXFactor, double newYFactor, double newXCenter, double newYCenter) {
    double newXRange = (xmax-xmin)/newXFactor;
    double newYRange = (ymax-ymin)/newYFactor;
    
    xmin = newXCenter - newXRange/2.0;
    ymin = newYCenter - newYRange/2.0;
    xmax = xmin + newXRange;
    ymax = ymin + newYRange;
    xscale *= newXFactor;
    yscale *= newYFactor;
    
    setTransform(xmin, xmax, ymin, ymax, xscale, yscale);
    }
  
	public void translateBy(double dx, double dy) {
		double newxmin = xmin + dx;
		double newymin = ymin + dy;
		setTransform(newxmin, newxmin + (xmax-xmin), newymin, newymin + (ymax-ymin), xscale, yscale);
		}
		
  public void centerAt(double xcenter, double ycenter) {
    double newxmin = xcenter - (xmax-xmin)/2.0;
    double newymin = ycenter - (ymax-ymin)/2.0;
    setTransform(newxmin, newxmin + (xmax-xmin), newymin, newymin + (ymax-ymin), xscale, yscale);
    }
  
  public double[] getPlotRange() {
    return new double[]{xmin, xmax, ymin, ymax};}
    
  public double[] getScale() {
    return new double[]{xscale, yscale};}
    
  public void setTransform(double xmin, double xmax, double ymin, double ymax,
      double xscale, double yscale) {
      
    this.xmin = xmin;
    this.xmax = xmax;
    this.ymin = ymin;
    this.ymax = ymax;

    this.xscale = xscale;
    this.yscale = yscale;

    if (axesFigure != null) {
      axesFigure.setTransform(xmin, ymax, xscale, yscale);
      }
    Iterator it = trekFigureHash.values().iterator();
    while(it.hasNext()) {
      ((TrekFigure)it.next()).recreate(xmin, xscale, ymax, yscale);
      }
    }

  public double[] getTrekCoordinate(LayerEvent e) {
    return new double[]{getTrekCoordinateX(e.getLayerX()), getTrekCoordinateY(e.getLayerY())};
    }
    
  protected double getTrekCoordinateX(double layerCoordX) {
    return xmin + layerCoordX/xscale;
    }

  protected double getTrekCoordinateY(double layerCoordY) {
    return ymax - layerCoordY/yscale;
    }

  public void updateTrekPoints() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    TrekEvent e = null;
    String[] keys = getTrekKeys();
    
    for (int i = 0; i < keys.length; ++i) {
      String thisKey = keys[i];
      e = null;
      for ( int j = lsns.length - 2; j >= 0; j -= 2 ) {
        if ( lsns[j] == TrekListener.class ) {
          if (e == null)
            e = new TrekEvent( thisKey, getTrekOrigin(thisKey), getTrekOriginIndependent(thisKey),
            			getTrekIndependentRange(thisKey), TrekEvent.ORIGIN_DID_CHANGE);
          ((TrekListener)lsns[j+1]).trekOriginDidChange(e);
          }
        }
      }
    }

  public void setSelectionColor(Color newColor) {
    SelectionModel sm = getTrekController().getSelectionModel();             
    ArrayList selKeys = new ArrayList();
    for(Iterator iter = sm.getSelection(); iter.hasNext();){
      String key = getTrekIdFromTarget((Figure)iter.next());
      if (key != null) {
        selKeys.add(key);
        }
      }
  
    for(Iterator iter = selKeys.iterator(); iter.hasNext();) {
      TrekFigure f = getTrekFigure((String)iter.next());
      if (f != null) f.setColor(newColor);
      }
      
    }
    
   public void setSelectionDisplayMode(int displayMode) {
    SelectionModel sm = getTrekController().getSelectionModel();             
    ArrayList selKeys = new ArrayList();
    for(Iterator iter = sm.getSelection(); iter.hasNext();){
      String key = getTrekIdFromTarget((Figure)iter.next());
      if (key != null) {
        selKeys.add(key);
        }
      }
  
    for(Iterator iter = selKeys.iterator(); iter.hasNext();) {
      TrekFigure f = getTrekFigure((String)iter.next());
      if (f != null) f.setDisplayMode(displayMode);
      }
      
    }
    
	public void setSelectionInitialConditions(double[] newInitials) {
		if (listeners == null) return;
		Object[] lsns = listeners.getListenerList();
		TrekEvent e = null;
		SelectionModel sm = getTrekController().getSelectionModel();             
		ArrayList selKeys = new ArrayList();
		for(Iterator iter = sm.getSelection(); iter.hasNext();){
			String key = getTrekIdFromTarget((Figure)iter.next());
			if (key != null) {
				selKeys.add(key);
				}
			}
	
		for(Iterator iter = selKeys.iterator(); iter.hasNext();) {
			String id = (String)iter.next();
			e = null;
			for ( int j = lsns.length - 2; j >= 0; j -= 2 ) {
				if ( lsns[j] == TrekListener.class ) {
					if (e == null) {
						e = new TrekEvent( id, newInitials, getTrekOriginIndependent(id), 
								getTrekIndependentRange(id), TrekEvent.ORIGIN_DID_CHANGE);
						}
					((TrekListener)lsns[j+1]).trekOriginDidChange(e);
					}
				}
			}
      
		}
  
	public void setSelectionOriginIndependent(double newVal) {
		if (listeners == null) return;
		Object[] lsns = listeners.getListenerList();
		TrekEvent e = null;
		SelectionModel sm = getTrekController().getSelectionModel();             
		ArrayList selKeys = new ArrayList();
		for(Iterator iter = sm.getSelection(); iter.hasNext();){
			String key = getTrekIdFromTarget((Figure)iter.next());
			if (key != null) {
				selKeys.add(key);
				}
			}
	
		for(Iterator iter = selKeys.iterator(); iter.hasNext();) {
			String id = (String)iter.next();
			e = null;
			for ( int j = lsns.length - 2; j >= 0; j -= 2 ) {
				if ( lsns[j] == TrekListener.class ) {
					if (e == null) {
						e = new TrekEvent( id, getTrekOrigin(id), newVal, 
								getTrekIndependentRange(id), TrekEvent.ORIGIN_DID_CHANGE);
						}
					((TrekListener)lsns[j+1]).trekOriginDidChange(e);
					}
				}
			}
      
		}
		
	public void setSelectionIndependentRange(double[] newIndependents) {
		if (listeners == null) return;
		Object[] lsns = listeners.getListenerList();
		TrekEvent e = null;
		SelectionModel sm = getTrekController().getSelectionModel();             
		ArrayList selKeys = new ArrayList();
		for(Iterator iter = sm.getSelection(); iter.hasNext();){
			String key = getTrekIdFromTarget((Figure)iter.next());
			if (key != null) {
				selKeys.add(key);
				}
			}
	
		for(Iterator iter = selKeys.iterator(); iter.hasNext();) {
			String id = (String)iter.next();
			e = null;
			for ( int j = lsns.length - 2; j >= 0; j -= 2 ) {
				if ( lsns[j] == TrekListener.class ) {
					if (e == null) {
						e = new TrekEvent( id, getTrekOrigin(id), getTrekOriginIndependent(id), newIndependents, TrekEvent.INDEPENDENT_RANGE_DID_CHANGE);
						}
					((TrekListener)lsns[j+1]).trekIndependentRangeDidChange(e);
					}
				}
			}
      
		}
  
		
  public void updateSelectedTrekPoints() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    TrekEvent e = null;
    Object[] targets = controller.getSelectionDragger().getTargetArray();
    
    for (int i = 0; i < targets.length; ++i) {
      Figure obj = (Figure)targets[i];
      e = null;
      for ( int j = lsns.length - 2; j >= 0; j -= 2 ) {
        if ( lsns[j] == TrekListener.class ) {
          if (e == null) {
          	String id = getTrekIdFromTarget(obj);
            e = new TrekEvent( id, getTargetCoordinate(obj), getTrekOriginIndependent(id),
            		getTrekIndependentRange(id), TrekEvent.ORIGIN_DID_CHANGE);
        		}
          ((TrekListener)lsns[j+1]).trekOriginDidChange(e);
          }
        }
      }
      
    }
  
  public String createTrekKey() {
    return "TrekObject" + (++uniqueID);
    }
  
  public void createdTrekAt(LayerEvent e) {
    String key = null;
    Figure f = e.getFigureSource();
    if (f != null && f instanceof TrekFigure) key = ((TrekFigure)f).getKey();
    else key = createTrekKey();
    fireTrekOriginChanged(key, getTrekCoordinate(e), getTrekOriginIndependent(key), getTrekIndependentRange(key));
    }
    
  public void draggedTrekAt(LayerEvent e) {
    String key = null;
    Figure f = e.getFigureSource();
    if (f != null && f instanceof TrekFigure) key = ((TrekFigure)f).getKey();
    else key = "TrekObject" + uniqueID;
    fireTrekOriginChanged(key, getTrekCoordinate(e), getTrekOriginIndependent(key), getTrekIndependentRange(key));
    }
    
  /**
   * Adds the specified TrekListener to receive TrekEvents.
   * <p>
   * Use this method to register a TrekListener object to receive
   * notifications when trek events occur
   *
   * @param l the TrekListener to register
   * @see #removeTrekListener(TrekListener)
   */
  public void addTrekListener(TrekListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( TrekListener.class, l );
      }
    }

  /**
   * Removes the specified TrekListener object so that it no longer receives
   * TrekEvents.
   *
   * @param l the TrekListener to register
   * @see #addTrekListener(TrekListener)
   */
  public void removeTrekListener(TrekListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( TrekListener.class, l );
      }
    }

  public void fireTrekOriginChanged(String key, double[] origin, double originIndep, double[] independentRange) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    TrekEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == TrekListener.class ) {
        if (e == null)
          e = new TrekEvent( key, origin, originIndep, independentRange, TrekEvent.ORIGIN_DID_CHANGE);
        ((TrekListener)lsns[i+1]).trekOriginDidChange(e);
        }
      }
    }
    
	public void fireTrekIndependentRangeChanged(String key, double[] origin, double originIndep, double[] independentRange) {
		if (listeners == null) return;
		Object[] lsns = listeners.getListenerList();
		TrekEvent e = null;

		// Process the listeners last to first, notifying
		// those that are interested in this event
		for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
			if ( lsns[i] == TrekListener.class ) {
				if (e == null)
					e = new TrekEvent( key, origin, originIndep, independentRange, TrekEvent.INDEPENDENT_RANGE_DID_CHANGE);
				((TrekListener)lsns[i+1]).trekIndependentRangeDidChange(e);
				}
			}
		}
		
}



