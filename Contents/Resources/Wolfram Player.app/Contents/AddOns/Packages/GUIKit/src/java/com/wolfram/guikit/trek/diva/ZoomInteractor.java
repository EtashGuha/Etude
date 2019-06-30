/*
 * $Id: ZoomInteractor.java,v 1.3 2004/04/02 22:47:45 jeffa Exp $
 *
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.event.EventLayer;
import diva.canvas.event.LayerEvent;
import diva.canvas.event.MouseFilter;
import diva.canvas.interactor.CompositeInteractor;

/** ZoomInteractor
 *
 * @version $Revision: 1.3 $
 */
public class ZoomInteractor extends CompositeInteractor {

    /** The mouse filter for selecting items
     */
    private MouseFilter _zoomInFilter = MouseFilter.selectionFilter;

    /** The mouse filter for toggling items
     */
    private MouseFilter _zoomOutFilter = MouseFilter.alternateSelectionFilter;

    private TrekPane trekPane;
    
    /* The event layer
     */
    private EventLayer _eventLayer;
    
    ///////////////////////////////////////////////////////////////////
    ////                         constructors                      ////

    /**
     * Create a new SelectionInteractor with a default selection model and
     * a default selection renderer.
     */
    public ZoomInteractor(TrekPane trekPane) {
      super();
      this.trekPane = trekPane;
      setEventLayer(trekPane.getBackgroundEventLayer());
      }

    ///////////////////////////////////////////////////////////////////
    //// public methods

    /**
     * Accept an event if it will be accepted by the selection
     * filters.
     */
    public boolean accept (LayerEvent e) {
  return _zoomInFilter.accept(e) || _zoomOutFilter.accept(e)
      || super.accept(e);
    }

    public MouseFilter getZoomInFilter () {
        return _zoomInFilter;
    }

    public MouseFilter getZoomOutFilter () {
        return _zoomOutFilter;
    }



  public void zoom(double centerX, double centerY, double zoomAmount) {
    // TODO first translate center to coords converted to trek coords then scale
    double newCenterX = trekPane.getTrekCoordinateX(centerX);
    double newCenterY = trekPane.getTrekCoordinateY(centerY);
    trekPane.addScaleFactor(zoomAmount, zoomAmount, newCenterX, newCenterY);
    }
    
    /** Handle a mouse press event. Add or remove the clicked-on
     * item to or from the selection. If it's still in the selection,
     * pass the event to the superclass to handle.
     */
    public void mousePressed (LayerEvent event) {
        if (!isEnabled()) {
            return;
        }
 
        boolean zoomed = false;
        
        if (_zoomInFilter.accept(event)) {
          zoomed = true;
          // TODO make the click zooming settable 200.0?
          zoom(event.getLayerX(), event.getLayerY(), 1.10);
          } 
        else if (_zoomOutFilter.accept(event)) {
          zoomed = true;
          zoom(event.getLayerX(), event.getLayerY(), 0.90);
          }

        // Allow superclass to process event
        super.mousePressed(event);

        // Always consume the event if the zoom occurred, regardless of the consuming flag
        if (zoomed) {
            event.consume();
        }
    }

    /**
     * Set the consuming flag of this interactor. This flag is a little
     * more complicated than in simple interactors: if not set, then
     * the event is consumed only if the clicked-on figure is added
     * to or removed from the selection. Otherwise it is not consumed.
     * If the flag is set, then the event is always consumed, thus
     * making it effectively "opaque" to events.
     *
     * <P> Note that the behaviour when the flag is false is the desired
     * behaviour when building panes that have an interactor attached
     * to the background. That way, the event passes through to the background
     * if a figure is hit on but the selection interactor's filters are
     * set up to ignore that particular event.
     *
     * <p> There is a third possibility, which is not supported: never
     * consume events. There is no way to do this currently, as the other
     * two behaviors seemed more likely to be useful. (Also, that behaviour
     * is harder to implement because of interaction with the superclass.)
     */
    public void setConsuming (boolean flag) {
        // This method is only here for documentation purposes
        super.setConsuming(flag);
    }

    public void setZoomInFilter(MouseFilter f) {
        _zoomInFilter = f;
    }

    public void setZoomOutFilter(MouseFilter f) {
        _zoomOutFilter = f;
    }
    
    /**
     * Set the layer that drag events are listened on
     */
    public void setEventLayer (EventLayer l) {
      if (_eventLayer != null) {
        _eventLayer.removeLayerListener(this);
        }
      _eventLayer = l;
      _eventLayer.addLayerListener(this);
      }
    
}



