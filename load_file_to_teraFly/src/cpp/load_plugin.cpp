/* load_plugin.cpp
 * This is a test plugin, you can use it as a demo.
 * 2023-11-24 : by crz
 */
 
#include "v3d_message.h"
#include <vector>
#include "load_plugin.h"
using namespace std;
Q_EXPORT_PLUGIN2(load, loadPlugin);
 
QStringList loadPlugin::menulist() const
{
	return QStringList() 
		<<tr("load_swc_to_teraFly")
		<<tr("load_mark_to_teraFly")
		<<tr("about");
}

QStringList loadPlugin::funclist() const
{
	return QStringList()
		<<tr("func1")
		<<tr("func2")
		<<tr("help");
}

void loadPlugin::domenu(const QString &menu_name, V3DPluginCallback2 &callback, QWidget *parent)
{
	if (menu_name == tr("load_swc_to_teraFly"))
	{
		auto dialog = new LoadSwcTeraFlyDialog(callback, parent);
        dialog->show();
		//v3d_msg("To be implemented1.");
	}
	else if (menu_name == tr("load_mark_to_teraFly"))
	{
		auto dialog = new LoadMarkTeraFlyDialog(callback,parent);
		dialog->show();
		//v3d_msg("To be implemented.");
	}
	else
	{
		v3d_msg(tr("This is a plugin for load files in teraFly"
			"Developed by crz, 2023-11-24"));
	}
}

bool loadPlugin::dofunc(const QString & func_name, const V3DPluginArgList & input, V3DPluginArgList & output, V3DPluginCallback2 & callback,  QWidget * parent)
{
	vector<char*> infiles, inparas, outfiles;
	if(input.size() >= 1) infiles = *((vector<char*> *)input.at(0).p);
	if(input.size() >= 2) inparas = *((vector<char*> *)input.at(1).p);
	if(output.size() >= 1) outfiles = *((vector<char*> *)output.at(0).p);

	if (func_name == tr("func1"))
	{
		v3d_msg("To be implemented.");
	}
	else if (func_name == tr("func2"))
	{
		v3d_msg("To be implemented.");
	}
	else if (func_name == tr("help"))
	{
		v3d_msg("To be implemented.");
	}
	else return false;

	return true;
}

