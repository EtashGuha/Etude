/*
 * @(#)TrekController.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.event.LayerAdapter;
import diva.canvas.event.LayerEvent;
import diva.canvas.event.MouseFilter;
import diva.canvas.interactor.*;

import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.InputEvent;
import java.awt.image.BufferedImage;
import java.beans.PropertyChangeListener;
import java.net.URL;

import javax.swing.ImageIcon;

import com.wolfram.guikit.diva.EllipseHighlighter;
import com.wolfram.guikit.diva.CompoundMouseFilter;

/**
 * TrekController handles the controller and filters associated with a TrekPane
 *
 * @version $Revision: 1.2 $
 */
public class TrekController  {

	public static final int MODE_SELECT = 1;
	public static final int MODE_CREATE = 2;
	public static final int MODE_ZOOM = 3;
	public static final int MODE_PAN = 4;
	
	private int mode = MODE_SELECT;
   
	public static final String MODE_PROPERTY = "mode";
	
  /**
   * The graphics pane that this is controlling.
   */
  private TrekPane _pane = null;

 /** The interactor for creating new nodes
   */
  private TrekCreator creationInteractor;
  
  /**
   * The default selection model.  Selected figures are
   * added to this selection model.
   */
  private SelectionModel _selectionModel = new BasicSelectionModel();

  /**
   * The interactor that all figures take. It enables figures to be
   * selected and moved around.
   */
  private DragInteractor dragger;
  private TrekSelectionInteractor selectionInteractor;
   /** The selection interactor for drag-selecting treks
   */
  private TrekSelectionDragger selectionDragger;
  
  private ZoomDragger zoomInteractor;
  private PanInteractor panInteractor;
  
  public static final Cursor SELECT_CURSOR = new Cursor(Cursor.DEFAULT_CURSOR);
	public static final Cursor CREATE_CURSOR = new Cursor(Cursor.CROSSHAIR_CURSOR);
	public static Cursor ZOOMIN_CURSOR;
	public static Cursor ZOOMOUT_CURSOR;
  // For cross-platform look we decide not to use MOVE_CURSOR
	public static Cursor PAN_CURSOR;
	
	private static final MouseFilter button1 = new MouseFilter(InputEvent.BUTTON1_MASK);
	private static final MouseFilter button1Shift = new MouseFilter(InputEvent.BUTTON1_MASK,InputEvent.SHIFT_MASK);
	private static final MouseFilter button1Control = new MouseFilter(InputEvent.BUTTON1_MASK, InputEvent.CTRL_MASK);
	private static final MouseFilter button1ShiftControl = new MouseFilter(InputEvent.BUTTON1_MASK, InputEvent.CTRL_MASK|InputEvent.SHIFT_MASK);
	private static final MouseFilter altButtons = new MouseFilter(InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK);
	private static final MouseFilter altButtonsShift = new MouseFilter(InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK, InputEvent.SHIFT_MASK);
	
	public static final MouseFilter primaryMouseFilter = new CompoundMouseFilter(new MouseFilter[]{
		button1, button1Shift });
	public static final MouseFilter primarySelectMouseFilter = button1;
	public static final MouseFilter primaryAlternateSelectMouseFilter = button1Shift;
	
	public static final MouseFilter secondaryMouseFilter = new CompoundMouseFilter(new MouseFilter[]{
		altButtons, altButtonsShift, button1Control, button1ShiftControl });
	public static final MouseFilter secondarySelectMouseFilter = new CompoundMouseFilter(new MouseFilter[]{
		altButtons, button1Control });
	public static final MouseFilter secondaryAlternateSelectMouseFilter = new CompoundMouseFilter(new MouseFilter[]{
		altButtonsShift, button1ShiftControl });
	
	private java.beans.PropertyChangeSupport changeSupport;
    
  private static Cursor createCustomCursor(String resource, int cursorHotspotX, int cursorHotspotY, String cursorName) {
    URL imageURL = TrekController.class.getClassLoader().getResource(resource);
    Image originalImage;
    if (imageURL == null) return null;
    originalImage = new ImageIcon(imageURL).getImage();
    int imWidth = originalImage.getWidth(null);
    int imHeight = originalImage.getHeight(null);

    Dimension bestSize = Toolkit.getDefaultToolkit().getBestCursorSize(imWidth, imHeight);
    Cursor cursor;
    if(bestSize.width == imWidth && bestSize.height == imHeight) {
      cursor  = Toolkit.getDefaultToolkit().createCustomCursor(originalImage,
           new Point(cursorHotspotX, cursorHotspotY),
           cursorName);
      } 
    else {
      BufferedImage cursorImage = new BufferedImage(bestSize.width, bestSize.height, BufferedImage.TYPE_INT_ARGB);
      Graphics2D g2 = cursorImage.createGraphics();
      g2.drawImage(originalImage, 0, 0, null);
      cursor  = Toolkit.getDefaultToolkit().createCustomCursor(cursorImage,
           new Point(cursorHotspotX, cursorHotspotY),
           cursorName);
       }
    return cursor;
    }
  
  static {
    //ZOOM_CURSOR = createCustomCursor("images/trek/zoom.gif", 4,4, "Zoom");
		ZOOMIN_CURSOR = createCustomCursor("images/trek/zoomin.gif", 4,4, "ZoomIn");
		ZOOMOUT_CURSOR = createCustomCursor("images/trek/zoomout.gif", 4,4, "ZoomOut");
    PAN_CURSOR = createCustomCursor("images/trek/pan.gif", 7,7, "Pan");
    } 
    
  public TrekController () {
    super();
    }

  public int getMode() {return mode;}
  public void setMode(int m) {
			if (m != mode) {
				int oldMode = mode;
				mode = m;
				// adjust the active filters and interactors based on mode (with secondary interactors)
			  switch (mode) {
			  	case MODE_SELECT:
					case MODE_CREATE:
			  	  zoomInteractor.setEnabled(false);
					  panInteractor.setEnabled(false);
					  dragger.setEnabled(true);
					  selectionInteractor.setEnabled(true);
					  selectionDragger.setEnabled(true);
					  creationInteractor.setEnabled(true);
					  dragger.setMouseFilter( mode == MODE_SELECT ? primaryMouseFilter : secondaryMouseFilter);
					  selectionDragger.setSelectionFilter(mode == MODE_SELECT ? primarySelectMouseFilter : secondarySelectMouseFilter);
						selectionDragger.setToggleFilter(mode == MODE_SELECT ? primaryAlternateSelectMouseFilter : secondaryAlternateSelectMouseFilter);
					  selectionInteractor.setSelectionFilter(mode == MODE_SELECT ? primarySelectMouseFilter : secondarySelectMouseFilter);
            selectionInteractor.setToggleFilter(mode == MODE_SELECT ? primaryAlternateSelectMouseFilter : secondaryAlternateSelectMouseFilter);
            creationInteractor.setMouseFilter(mode == MODE_CREATE ? primaryMouseFilter : secondaryMouseFilter);
					  _pane.getCanvas().setCursor(mode == MODE_SELECT ? SELECT_CURSOR : CREATE_CURSOR);
			      break;
			  	case MODE_ZOOM:
			    case MODE_PAN:
					  zoomInteractor.setEnabled(true);
						panInteractor.setEnabled(true);
						dragger.setEnabled(false);
						selectionInteractor.setEnabled(false);
						selectionDragger.setEnabled(false);
						creationInteractor.setEnabled(false);
					  zoomInteractor.setZoomInFilter( mode == MODE_ZOOM ? primarySelectMouseFilter : secondarySelectMouseFilter);
            zoomInteractor.setZoomOutFilter( mode == MODE_ZOOM ? primaryAlternateSelectMouseFilter : secondaryAlternateSelectMouseFilter);
					  panInteractor.setPanFilter(mode == MODE_PAN ? primarySelectMouseFilter : secondarySelectMouseFilter);
					  _pane.getCanvas().setCursor(mode == MODE_ZOOM ? ZOOMIN_CURSOR : PAN_CURSOR);
			    	break;
			  	  }
				firePropertyChange(MODE_PROPERTY, new Integer(oldMode), new Integer(mode));
			  }
			}
		
		public void setToggleControls(boolean isControlDown, boolean isShiftDown) {
			switch (mode) {
				case MODE_SELECT:
				  _pane.getCanvas().setCursor(isControlDown ? CREATE_CURSOR : SELECT_CURSOR);
					break;
				case MODE_CREATE:
				  _pane.getCanvas().setCursor(isControlDown ? SELECT_CURSOR : CREATE_CURSOR);
					break;
				case MODE_ZOOM:
					_pane.getCanvas().setCursor((isControlDown && !zoomInteractor.isMouseDown()) ? PAN_CURSOR : (isShiftDown ? ZOOMOUT_CURSOR : ZOOMIN_CURSOR));
					break;
				case MODE_PAN:
				  _pane.getCanvas().setCursor(isControlDown ? (isShiftDown ? ZOOMOUT_CURSOR : ZOOMIN_CURSOR) : PAN_CURSOR);
					break;
			  }
			}
			
    /**
     * Return the selection interactor on all symbols.
     */
    public DragInteractor getSelectionDragger() {return dragger;}

    /**
     * Return the selection interactor on all symbols.
     */
    public TrekCreator getCreationInteractor(){ return creationInteractor;}
    
    /**
     * Return the selection interactor on all symbols.
     */
    public Interactor getSelectionInteractor(){ 
      return selectionInteractor;
      }

    /**
     * Get the default selection model.
     */
    public SelectionModel getSelectionModel () {return _selectionModel;}

    /**
     * Return the parent pane of this controller.
     */
    public TrekPane getTrekPane() {return _pane;}

    /**
     * Initialize all interaction on the Trek pane.  This method is
     * called by the setTrekPane() method.  The initialization
     * cannot be done in the constructor because the controller does
     * not yet have a reference to its pane at that time.
     */
    protected void initializeInteraction () {
        SelectionModel sm = getSelectionModel();
        final TrekPane pane = getTrekPane();

        // Selection interactor for selecting objects
        dragger = new DragInteractor();
        dragger.setMouseFilter(primaryMouseFilter);
        dragger.setSelectiveEnabled(true);
        /* This listener notifies TrekPane of the need to update or
         * create trek points for a trek when selected treks are moved */
        dragger.addLayerListener(new LayerAdapter() {
            public void mouseDragged(LayerEvent e) {
               // Selected treks have moved so notify TrekPane
               // to request new coordinates
               ((TrekPane)pane).updateSelectedTrekPoints();
               }
             }
           );
        
        /** The selection interactor enabled figures to be selected
         * using conventional click-selection and drag-selection
         */
        selectionInteractor = new TrekSelectionInteractor(pane, sm);
        selectionInteractor.setConsuming(false);
        selectionInteractor.setPrototypeDecorator(new EllipseHighlighter());
        selectionInteractor.addInteractor(dragger);
        
        // Create and set up a selection dragger
        // Turn this on when we can drag out a rect and include
        // selectionFigures of treks.. think about easiest way or
        // doing a subclass
        //ok to turn on??
        selectionDragger = new TrekSelectionDragger(pane);
			  selectionDragger.setSelectionFilter(primarySelectMouseFilter);
			  selectionDragger.setToggleFilter(primaryAlternateSelectMouseFilter);
			  selectionDragger.setConsuming(false);
        selectionDragger.addSelectionInteractor(selectionInteractor);


        // Create a listener that creates new nodes
			  creationInteractor = new TrekCreator(pane);
			  creationInteractor.setConsuming(false);
			  creationInteractor.setMouseFilter(secondaryMouseFilter);
        
        /* This listener notifies TrekPane of the need to update or
         * create trek points for a trek */
			  creationInteractor.addLayerListener(new LayerAdapter() {
            public void mouseDragged(LayerEvent e) {
               ((TrekPane)pane).draggedTrekAt(e);
               }
            public void mousePressed(LayerEvent e) {
               ((TrekPane)pane).createdTrekAt(e);
               }
             }
           );
        
        panInteractor = new PanInteractor(pane);
			  panInteractor.setConsuming(false);
			  panInteractor.setEnabled(false);
			  
        zoomInteractor = new ZoomDragger(pane);
			  zoomInteractor.setConsuming(false);
			  zoomInteractor.setEnabled(false);
			  
        pane.getBackgroundEventLayer().addInteractor(creationInteractor);
        
      }

    /**
     * Set the default selection model. The caller is expected to ensure
     * that the old model is empty before calling this.
     */
    public void setSelectionModel(SelectionModel m){ _selectionModel = m;}

    /**
     * Set the graphics pane.  This is done once by the SketchPane
     * immediately after construction of the controller.
     */
    public void setTrekPane(TrekPane p){
      _pane = p;
      initializeInteraction();
      }
      
	/**
	 * Adds a PropertyChangeListener to the listener list. The listener is
	 * registered for all bound properties of this class, including the
	 * following:
	 * <ul>
	 *    <li>this Component's font ("font")</li>
	 *    <li>this Component's background color ("background")</li>
	 *    <li>this Component's foreground color ("foreground")</li>
	 *    <li>this Component's focusability ("focusable")</li>
	 *    <li>this Component's focus traversal keys enabled state
	 *        ("focusTraversalKeysEnabled")</li>
	 *    <li>this Component's Set of FORWARD_TRAVERSAL_KEYS
	 *        ("forwardFocusTraversalKeys")</li>
	 *    <li>this Component's Set of BACKWARD_TRAVERSAL_KEYS
	 *        ("backwardFocusTraversalKeys")</li>
	 *    <li>this Component's Set of UP_CYCLE_TRAVERSAL_KEYS
	 *        ("upCycleFocusTraversalKeys")</li>
	 * </ul>
	 * Note that if this Component is inheriting a bound property, then no
	 * event will be fired in response to a change in the inherited property.
	 * <p>
	 * If listener is null, no exception is thrown and no action is performed.
	 *
	 * @param    listener  the PropertyChangeListener to be added
	 *
	 * @see #removePropertyChangeListener
	 * @see #getPropertyChangeListeners
	 * @see #addPropertyChangeListener(java.lang.String, java.beans.PropertyChangeListener)
	 */
	public synchronized void addPropertyChangeListener(
													 PropertyChangeListener listener) {
			if (listener == null) {
					return;
			}
			if (changeSupport == null) {
					changeSupport = new java.beans.PropertyChangeSupport(this);
			}
			changeSupport.addPropertyChangeListener(listener);
		}
  
	/**
	 * Removes a PropertyChangeListener from the listener list. This method
	 * should be used to remove PropertyChangeListeners that were registered
	 * for all bound properties of this class.
	 * <p>
	 * If listener is null, no exception is thrown and no action is performed.
	 *
	 * @param listener the PropertyChangeListener to be removed
	 *
	 * @see #addPropertyChangeListener
	 * @see #getPropertyChangeListeners
	 * @see #removePropertyChangeListener(java.lang.String,java.beans.PropertyChangeListener)
	 */
	public synchronized void removePropertyChangeListener(
													 PropertyChangeListener listener) {
		if (listener == null || changeSupport == null) {
			return;
			}
		changeSupport.removePropertyChangeListener(listener);
		}

	/**
	 * Returns an array of all the property change listeners
	 * registered on this component.
	 *
	 * @return all of this component's <code>PropertyChangeListener</code>s
	 *         or an empty array if no property change
	 *         listeners are currently registered
	 *
	 * @see      #addPropertyChangeListener
	 * @see      #removePropertyChangeListener
	 * @see      #getPropertyChangeListeners(java.lang.String)
	 * @see      java.beans.PropertyChangeSupport#getPropertyChangeListeners
	 * @since    1.4
	 */
	public synchronized PropertyChangeListener[] getPropertyChangeListeners() {
			if (changeSupport == null) {
					return new PropertyChangeListener[0];
				}
			return changeSupport.getPropertyChangeListeners();
			}
  
	/**
	 * Adds a PropertyChangeListener to the listener list for a specific
	 * property. The specified property may be user-defined, or one of the
	 * following:
	 * <ul>
	 *    <li>this Component's font ("font")</li>
	 *    <li>this Component's background color ("background")</li>
	 *    <li>this Component's foreground color ("foreground")</li>
	 *    <li>this Component's focusability ("focusable")</li>
	 *    <li>this Component's focus traversal keys enabled state
	 *        ("focusTraversalKeysEnabled")</li>
	 *    <li>this Component's Set of FORWARD_TRAVERSAL_KEYS
	 *        ("forwardFocusTraversalKeys")</li>
	 *    <li>this Component's Set of BACKWARD_TRAVERSAL_KEYS
	 *        ("backwardFocusTraversalKeys")</li>
	 *    <li>this Component's Set of UP_CYCLE_TRAVERSAL_KEYS
	 *        ("upCycleFocusTraversalKeys")</li>
	 * </ul>
	 * Note that if this Component is inheriting a bound property, then no
	 * event will be fired in response to a change in the inherited property.
	 * <p>
	 * If listener is null, no exception is thrown and no action is performed.
	 *
	 * @param propertyName one of the property names listed above
	 * @param listener the PropertyChangeListener to be added
	 *
	 * @see #removePropertyChangeListener(java.lang.String, java.beans.PropertyChangeListener)
	 * @see #getPropertyChangeListeners(java.lang.String)
	 * @see #addPropertyChangeListener(java.lang.String, java.beans.PropertyChangeListener)
	 */
	public synchronized void addPropertyChangeListener(
													 String propertyName,
													 PropertyChangeListener listener) {
		if (listener == null) {
			return;
			}
		if (changeSupport == null) {
			changeSupport = new java.beans.PropertyChangeSupport(this);
			}
		changeSupport.addPropertyChangeListener(propertyName, listener);
		}

	/**
	 * Removes a PropertyChangeListener from the listener list for a specific
	 * property. This method should be used to remove PropertyChangeListeners
	 * that were registered for a specific bound property.
	 * <p>
	 * If listener is null, no exception is thrown and no action is performed.
	 *
	 * @param propertyName a valid property name
	 * @param listener the PropertyChangeListener to be removed
	 *
	 * @see #addPropertyChangeListener(java.lang.String, java.beans.PropertyChangeListener)
	 * @see #getPropertyChangeListeners(java.lang.String)
	 * @see #removePropertyChangeListener(java.beans.PropertyChangeListener)
	 */
	public synchronized void removePropertyChangeListener(
													 String propertyName,
													 PropertyChangeListener listener) {
			if (listener == null || changeSupport == null) {
					return;
					}
			changeSupport.removePropertyChangeListener(propertyName, listener);
			}

	/**
	 * Returns an array of all the listeners which have been associated 
	 * with the named property.
	 *
	 * @return all of the <code>PropertyChangeListeners</code> associated with
	 *         the named property or an empty array if no listeners have 
	 *         been added
	 *
	 * @see #addPropertyChangeListener(java.lang.String, java.beans.PropertyChangeListener)
	 * @see #removePropertyChangeListener(java.lang.String, java.beans.PropertyChangeListener)
	 * @see #getPropertyChangeListeners
	 * @since 1.4
	 */
	public synchronized PropertyChangeListener[] getPropertyChangeListeners(
																											String propertyName) {
			if (changeSupport == null) {
					return new PropertyChangeListener[0];
				}
			return changeSupport.getPropertyChangeListeners(propertyName);
			}	

	/**
	 * Support for reporting bound property changes for Object properties. 
	 * This method can be called when a bound property has changed and it will
	 * send the appropriate PropertyChangeEvent to any registered
	 * PropertyChangeListeners.
	 *
	 * @param propertyName the property whose value has changed
	 * @param oldValue the property's previous value
	 * @param newValue the property's new value
	 */
	protected void firePropertyChange(String propertyName,
						Object oldValue, Object newValue) {
			java.beans.PropertyChangeSupport changeSupport = this.changeSupport;
		if (changeSupport == null) {
				return;
				}
		changeSupport.firePropertyChange(propertyName, oldValue, newValue);
		}

	/**
	 * Support for reporting bound property changes for boolean properties. 
	 * This method can be called when a bound property has changed and it will
	 * send the appropriate PropertyChangeEvent to any registered
	 * PropertyChangeListeners.
	 *
	 * @param propertyName the property whose value has changed
	 * @param oldValue the property's previous value
	 * @param newValue the property's new value
	 */
	protected void firePropertyChange(String propertyName,
																	boolean oldValue, boolean newValue) {
			java.beans.PropertyChangeSupport changeSupport = this.changeSupport;
			if (changeSupport == null) {
					return;
			}
		changeSupport.firePropertyChange(propertyName, oldValue, newValue);
		}
  
	/**
	 * Support for reporting bound property changes for integer properties. 
	 * This method can be called when a bound property has changed and it will
	 * send the appropriate PropertyChangeEvent to any registered
	 * PropertyChangeListeners.
	 *
	 * @param propertyName the property whose value has changed
	 * @param oldValue the property's previous value
	 * @param newValue the property's new value
	 */
	protected void firePropertyChange(String propertyName,
						int oldValue, int newValue) {
			java.beans.PropertyChangeSupport changeSupport = this.changeSupport;
		if (changeSupport == null) {
			return;
			}
		changeSupport.firePropertyChange(propertyName, oldValue, newValue);
		}
	
}



