/* singel_neuron_reconstruction_plugin.cpp
 * This is a test plugin, you can use it as a demo.
 * 2024-7-22 : by Runze Chen
 */
 
#include "v3d_message.h"
#include <vector>
#include "singel_neuron_reconstruction_plugin.h"
using namespace std;
Q_EXPORT_PLUGIN2(singel_neuron_reconstruction, TestPlugin);
 
QStringList TestPlugin::menulist() const
{
	return QStringList() 
		<<tr("reconstruction set")
		<<tr("about");
}

QStringList TestPlugin::funclist() const
{
	return QStringList()
		<<tr("func1")
		<<tr("func2")
		<<tr("help");
}

void TestPlugin::domenu(const QString &menu_name, V3DPluginCallback2 &callback, QWidget *parent)
{
	if (menu_name == tr("reconstruction set"))
	{
		auto dialog = new Reconstruction_Dialog(callback,parent);
		dialog->exec();
		//v3d_msg("To be implemented.");
	}
	else if (menu_name == tr("menu2"))
	{
		v3d_msg("To be implemented.");
	}
	else
	{
		v3d_msg(tr("This is a test plugin, you can use it as a demo.. "
			"Developed by Runze Chen, 2024-7-22"));
	}
}

bool TestPlugin::dofunc(const QString & func_name, const V3DPluginArgList & input, V3DPluginArgList & output, V3DPluginCallback2 & callback,  QWidget * parent)
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
