/*
 * @(#)DefaultFontSelectionModel.java	1.14 03/01/23
 */
package com.wolfram.guikit.swing;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.EventListenerList;
import java.awt.Font;
import java.io.Serializable;

/**
 * A generic implementation of <code>FontSelectionModel</code>.
 *
 * @version 1.0
 * @see java.awt.Font
 */
public class DefaultFontSelectionModel implements FontSelectionModel, Serializable {

  private static final long serialVersionUID = -1287947975454788448L;
  
    /**
     * Only one <code>ChangeEvent</code> is needed per model instance
     * since the event's only (read-only) state is the source property.
     * The source of events generated here is always "this".
     */
    protected transient ChangeEvent changeEvent = null;

    protected EventListenerList listenerList = new EventListenerList();

    private Font selectedFont;

    private static final Font DEFAULT_FONT = new Font("Serif", Font.PLAIN, 12);

    /**
     * Creates a <code>DefaultFontSelectionModel</code> with the
     * current font set to <code>Dialog</code>.  This is
     * the default constructor.
     */
    public DefaultFontSelectionModel() {
      selectedFont = DEFAULT_FONT;
      }

    /**
     * Creates a <code>DefaultFontSelectionModel</code> with the
     * current font set to <code>font</code>, which should be
     * non-<code>null</code>.  Note that setting the font to
     * <code>null</code> is undefined and may have unpredictable
     * results.
     *
     * @param font the new <code>Font</code>
     */
    public DefaultFontSelectionModel(Font font) {
      selectedFont = font;
      }

    /**
     * Returns the selected <code>Font</code> which should be
     * non-<code>null</code>.
     *
     * @return the selected <code>Font</code>
     */
    public Font getSelectedFont() {
      return selectedFont;
      }

    /**
     * Sets the selected font to <code>font</code>.
     * Note that setting the font to <code>null</code> 
     * is undefined and may have unpredictable results.
     * This method fires a state changed event if it sets the
     * current font to a new non-<code>null</code> font;
     * if the new font is the same as the current font,
     * no event is fired.
     *
     * @param font the new <code>Font</code>
     */
    public void setSelectedFont(Font font) {
      if (font != null && !selectedFont.equals(font)) {
        selectedFont = font;
        fireStateChanged();
        }
      }


    /**
     * Adds a <code>ChangeListener</code> to the model.
     *
     * @param l the <code>ChangeListener</code> to be added
     */
    public void addChangeListener(ChangeListener l) {
	   listenerList.add(ChangeListener.class, l);
      }

    /**
     * Removes a <code>ChangeListener</code> from the model.
     * @param l the <code>ChangeListener</code> to be removed
     */
    public void removeChangeListener(ChangeListener l) {
      listenerList.remove(ChangeListener.class, l);
      }

    /**
     * Returns an array of all the <code>ChangeListener</code>s added
     * to this <code>DefaultColorSelectionModel</code> with
     * <code>addChangeListener</code>.
     *
     * @return all of the <code>ChangeListener</code>s added, or an empty
     *         array if no listeners have been added
     * @since 1.4
     */
    public ChangeListener[] getChangeListeners() {
      return (ChangeListener[])listenerList.getListeners(
        ChangeListener.class);
      }

    /**
     * Runs each <code>ChangeListener</code>'s
     * <code>stateChanged</code> method.
     *
     * <!-- @see #setRangeProperties    //bad link-->
     * @see EventListenerList
     */
    protected void fireStateChanged() {
      Object[] listeners = listenerList.getListenerList();
      for (int i = listeners.length - 2; i >= 0; i -=2 ) {
        if (listeners[i] == ChangeListener.class) {
          if (changeEvent == null) {
            changeEvent = new ChangeEvent(this);
            }
          ((ChangeListener)listeners[i+1]).stateChanged(changeEvent);
          }
        }
      }

}
