package com.wolfram.databaselink.examples;

import java.awt.Color;

import javax.swing.JTable;

import com.odellengineeringltd.glazedlists.jtable.*;

public class CommandTableCell implements TableFormat
{
  private StripedTableCellRenderer stcr;
  private Color oddRowsColor = Color.WHITE;
  private Color evenRowsColor = new Color(221, 221, 255);

  public CommandTableCell()
  {
    stcr = new StripedTableCellRenderer(oddRowsColor, evenRowsColor, new CommandRenderer());
  }

  public int getFieldCount()
  {
    return 1;
  }

  public String getFieldName(int column)
  {
    return "Command";
  }

  public Object getFieldValue(Object baseObject, int column)
  {
    Command command = (Command)baseObject;
    return command;
  }

  public void configureTable(JTable table)
  {
    table.setGridColor(Color.WHITE);
    table.getColumnModel().getColumn(0).setCellRenderer(stcr);
    table.setTableHeader(null);
  }

}
